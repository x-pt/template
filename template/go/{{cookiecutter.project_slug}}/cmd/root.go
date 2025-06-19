// Package cmd implements the command-line interface for the {{cookiecutter.project_slug}} application.
// It uses the Cobra library to create a CLI application structure, supporting commands,
// subcommands, flags, and configuration management with Viper.
package cmd

import (
	"fmt"     // For formatted I/O (printing messages)
	"os"      // For exiting the application (os.Exit)
	"strings" // For string manipulations, e.g., environment variable prefixes

	// Cobra is a CLI library for Go that empowers applications.
	// This application uses Cobra for command handling, flag parsing, etc.
	"github.com/spf13/cobra"
	// Viper is a complete configuration solution for Go applications
	// including 12-Factor apps. It supports JSON, TOML, YAML, HCL, env vars and more.
	"github.com/spf13/viper"
)

// cfgFile is a package-level variable to store the path to the config file,
// potentially set by a command-line flag.
var cfgFile string

// rootCmd represents the base command when called without any subcommands.
// It's the entry point for the CLI application.
var rootCmd = &cobra.Command{
	Use:   "{{cookiecutter.project_slug}}", // This is how the command is invoked
	Short: "{{cookiecutter.project_desc}}", // A short description shown in help text
	Long: `A longer description for {{cookiecutter.project_slug}}.
This application can serve as a starting point for various Go projects.
It demonstrates:
  - Cobra for CLI structure (commands, flags)
  - Viper for configuration management (file, env vars, defaults)
  - A sample 'version' subcommand.
  - Basic greeting functionality.`, // A longer description shown in help text
	// The Run function is executed when the rootCmd is called.
	Run: func(cmd *cobra.Command, args []string) {
		// cmd.Use provides the command name.
		// {{cookiecutter.project_version}} is templated from cookiecutter.json.
		fmt.Printf("Hello from %s version %s!\n", cmd.Use, "{{cookiecutter.project_version}}")
		fmt.Println("This is a sample CLI application generated from the template.")

		// Demonstrate reading from Viper config (with defaults set in setDefaultConfig).
		// Viper attempts to find these keys in the config file, environment variables, or uses defaults.
		author := viper.GetString("app.author")
		if author != "" {
			fmt.Printf("Default author from config/defaults: %s\n", author)
		}
		description := viper.GetString("app.description")
		if description != "" {
			fmt.Printf("Description: %s\n", description)
		}

		fmt.Println("\nTry '--help' for more options, or try the 'version' subcommand.")
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
// It's the main entry point for Cobra execution.
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		// Cobra already prints the error to stderr, so no need for fmt.Println(err) here.
		os.Exit(1) // Exit with a non-zero status to indicate failure.
	}
}

// init is a special Go function that is called automatically when the package is initialized.
// It's used here to set up Cobra command initialization and persistent flags.
func init() {
	// cobra.OnInitialize registers functions to be called when Cobra is initializing.
	// initConfig is called here to load configuration before commands are executed.
	cobra.OnInitialize(initConfig)

	// Persistent flags are global for the application; they are available to the
	// root command and all its subcommands.
	// Here, we define a --config flag to specify an alternative config file.
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.{{cookiecutter.project_slug}}.yaml or ./config.yaml)")

	// Example of a local flag for the root command (only available for rootCmd itself).
	// rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}

// initConfig reads in config file and environment variables if set.
// It's called by cobra.OnInitialize.
func initConfig() {
	// --- Set Default Configuration Values ---
	// Defaults are set first, so they are overridden by config file, then env vars.
	setDefaultConfig()

	if cfgFile != "" {
		// Use config file from the --config flag.
		viper.SetConfigFile(cfgFile)
		// fmt.Printf("Using specific config file from flag: %s\n", cfgFile)
	} else {
		// Search for config file in standard locations.
		// 1. User's home config directory.
		home, err := os.UserHomeDir()
		cobra.CheckErr(err) // Exits on error, part of Cobra's error handling.
		viper.AddConfigPath(home + "/.config/{{cookiecutter.project_slug}}") // Path for user-specific config
		// 2. Current working directory.
		viper.AddConfigPath(".")
		// 3. A 'config' subdirectory in the current working directory (if any).
		viper.AddConfigPath("./config/")

		viper.SetConfigName("config") // Name of config file (without extension).
		viper.SetConfigType("yaml")   // Type of config file. Viper supports json, toml, yaml, hcl, env, etc.
		// fmt.Println("Searching for 'config.yaml' in standard locations...")
	}

	// --- Environment Variable Handling ---
	// Read in environment variables that match defined keys.
	// e.g., if a key is "app.author", an env var APP_AUTHOR will override it.
	viper.SetEnvPrefix("{{cookiecutter.project_slug | upper}}") // Prefix for environment variables, e.g., MYAWESOMEPROJECT_
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))    // Replaces dots with underscores for env var names.
	viper.AutomaticEnv()                                      // Automatically read matching env variables.

	// --- Read Configuration File ---
	// If a config file is found (based on SetConfigFile or search paths), read it in.
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			// Config file not found; ignore this error as defaults and env vars will be used.
			// It's not an error if the config file is optional.
			// fmt.Println("Warning: Configuration file not found. Using defaults and/or environment variables.")
		} else {
			// Config file was found but another error was produced (e.g., malformed YAML/JSON).
			// This is a more serious error.
			fmt.Printf("Error reading config file '%s': %s\n", viper.ConfigFileUsed(), err)
			// os.Exit(1) // Optionally exit if config is mandatory and unreadable.
		}
	} else {
		// Config file successfully loaded.
		// fmt.Printf("Using config file: %s\n", viper.ConfigFileUsed())
	}
}

// setDefaultConfig sets default values for configuration parameters.
// These defaults are used if the values are not found in the config file or environment variables.
func setDefaultConfig() {
	viper.SetDefault("app.author", "{{cookiecutter.full_name}}")
	viper.SetDefault("app.description", "{{cookiecutter.project_desc}}")
	// Add more defaults here, e.g.:
	// viper.SetDefault("database.host", "localhost")
	// viper.SetDefault("database.port", 5432)
}
