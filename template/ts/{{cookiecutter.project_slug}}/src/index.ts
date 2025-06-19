// src/index.ts
// This is the main entry point for the application.
// It demonstrates command-line argument parsing using 'yargs' and modular code structure.

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers'; // Utility to hide node and script name from argv
import { generateGreeting } from './greeting'; // Import from local module

// Define an interface for the expected shape of parsed arguments for type safety.
interface Arguments {
    name: string;
    verbose: boolean;
    // Add other expected arguments here
    [x: string]: unknown; // Allow for other arguments not explicitly defined
}

/**
 * Main asynchronous function to run the application.
 * It configures yargs for command-line argument parsing,
 * processes the arguments, and then prints a greeting.
 */
async function main(): Promise<void> {
    // Configure yargs to parse command-line arguments.
    // 'hideBin(process.argv)' removes the first two elements (node executable, script path)
    // from the arguments array, so yargs only sees the actual script arguments.
    const argv = await yargs(hideBin(process.argv))
        .option('name', { // Define a 'name' option
            alias: 'n', // Short alias '-n'
            type: 'string', // Expected type
            description: 'Your name for the greeting',
            default: 'World', // Default value if not provided
        })
        .option('verbose', { // Define a 'verbose' option
            alias: 'v', // Short alias '-v'
            type: 'boolean', // Expected type
            description: 'Run with verbose logging',
            default: false, // Default value if not provided
        })
        .help() // Enable the default --help option
        .alias('help', 'h') // Alias -h for --help
        .version(false) // Disable yargs' default --version flag as we might add a custom one.
        // .strict() // Uncomment to enable strict mode (error on unknown options)
        .parseAsync() as Arguments; // Parse arguments and cast to our interface

    // If verbose mode is enabled, print the parsed arguments.
    if (argv.verbose) {
        console.log('Verbose mode enabled. Arguments received:');
        // Use JSON.stringify for a clean print of the argv object.
        console.log(JSON.stringify(argv, null, 2));
    }

    // Generate the greeting message using the imported function and the 'name' argument.
    const message = generateGreeting(argv.name);
    console.log(message); // Print the greeting.

    // Provide a hint if the default name was used.
    if (argv.name === 'World') {
        console.info("\nHint: Try running with '--name YourName' or '-n YourName' to personalize the greeting.");
    }
}

// Execute the main function.
// The 'JEST_WORKER_ID' environment variable is typically set by Jest when running tests.
// This check prevents the main function from automatically executing during test runs,
// allowing tests to call main() or other functions explicitly if needed.
if (process.env.JEST_WORKER_ID === undefined) {
    main().catch(error => {
        // Catch and log any unhandled errors from the main function.
        console.error('An unexpected error occurred in the application:');
        console.error(error);
        process.exit(1); // Exit with a non-zero status code to indicate failure.
    });
}
