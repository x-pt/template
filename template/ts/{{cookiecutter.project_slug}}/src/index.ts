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

async function checkAPIReachability(apiUrl: string): Promise<void> {
    try {
        const response = await axios.get(apiUrl, { timeout: 10000 });
        if (response.status < 200 || response.status >= 300) {
            core.warning(`API is not reachable, status code: ${response.status}`);
        } else {
            core.info(`API ${apiUrl} is reachable.`);
        }
    } catch (error) {
        core.warning(`Failed to make API request: ${error instanceof Error ? error.message : String(error)}`);
    }
}

async function readAndAppendToFile(inputFile: string, outputFile: string, appendText: string): Promise<void> {
    try {
        const content = await fs.readFile(inputFile, "utf-8");
        const modifiedContent = `${content}\n${appendText}`;
        await fs.writeFile(outputFile, modifiedContent, { encoding: "utf-8" });
        core.info(`Appended text to file: ${outputFile}`);
    } catch (error) {
        throw new Error(`File operation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
}

function processText(
    text: string,
    findWord: string,
    replaceWord: string,
): {
    processedText: string;
    wordCount: number;
} {
    const processedText = text.replace(new RegExp(findWord, "g"), replaceWord);
    const wordCount = processedText.trim() === "" ? 0 : processedText.trim().split(/\s+/).length;
    return { processedText, wordCount };
}

function calculateNumberStats(numbers: number[]): { sum: number; average: number } {
    const sum = numbers.reduce((acc, num) => acc + num, 0);
    const average = numbers.length > 0 ? sum / numbers.length : 0;
    return { sum, average };
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
            await checkAPIReachability(api_url);
        }

        if (input_file && output_file && append_text) {
            await readAndAppendToFile(input_file, output_file, append_text);
        }

        const { processedText, wordCount } = processText(input_text, find_word, replace_word);
        const { sum, average } = calculateNumberStats(number_list);

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
