import * as fs from "node:fs/promises";
import * as core from "@actions/core";
import axios from "axios";
import * as toml from "toml";

interface Config {
    input_text?: string;
    find_word?: string;
    replace_word?: string;
    number_list?: number[];
    input_file?: string;
    output_file?: string;
    append_text?: string;
    api_url?: string;
}

// Function to check API reachability
async function checkAPIReachability(apiUrl: string): Promise<void> {
    try {
        const response = await axios.get(apiUrl, { timeout: 10000 });
        if (response.status < 200 || response.status >= 300) {
            core.warning(`API is not reachable, status code: ${response.status}`);
        }
    } catch (error) {
        core.warning(`Failed to make API request: ${error instanceof Error ? error.message : String(error)}`);
    }
}

// Function to read and append text to a file
async function readAndAppendToFile(inputFile: string, outputFile: string, appendText: string): Promise<void> {
    try {
        const content = await fs.readFile(inputFile, "utf-8");
        const modifiedContent = `${content}\n${appendText}`;
        await fs.writeFile(outputFile, modifiedContent, { encoding: "utf-8" });
    } catch (error) {
        throw new Error(`File operation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
}

export async function run(): Promise<void> {
    try {
        const configPath = core.getInput("config_path") || ".github/configs/setup-custom-action-by-ts.toml";

        const configContent = await fs.readFile(configPath, "utf-8");
        const config: Config = toml.parse(configContent);

        const {
            input_text = "",
            find_word = "",
            replace_word = "",
            number_list = [],
            input_file = "",
            output_file = "",
            append_text = "",
            api_url = "",
        } = config;

        if (api_url) {
            try {
                await checkAPIReachability(api_url);
                core.info(`API ${api_url} is reachable.`);
            } catch (error) {
                core.warning(error instanceof Error ? error.message : String(error));
            }
        }

        if (input_file && output_file && append_text) {
            try {
                await readAndAppendToFile(input_file, output_file, append_text);
                core.info(`Appended text to file: ${output_file}`);
            } catch (error) {
                core.warning(error instanceof Error ? error.message : String(error));
            }
        }

        const processedText = input_text.replace(new RegExp(find_word, "g"), replace_word);
        const wordCount = processedText.trim() === "" ? 0 : processedText.trim().split(/\s+/).length;

        const sum = number_list.reduce((acc: number, num: number) => acc + num, 0);
        const average = number_list.length > 0 ? sum / number_list.length : 0;

        core.setOutput("processed_text", processedText);
        core.setOutput("word_count", wordCount);
        core.setOutput("sum", sum);
        core.setOutput("average", average);
    } catch (error) {
        core.setFailed(`Action failed with error: ${error instanceof Error ? error.message : String(error)}`);
    }
}

if (!process.env.JEST_WORKER_ID) {
    run().catch((error) => {
        console.error("Unhandled error:", error);
        process.exit(1);
    });
}
