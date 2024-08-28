import * as fs from "node:fs/promises";
import * as core from "@actions/core";
import axios from "axios";
import { run } from "../src";

jest.mock("@actions/core");
jest.mock("node:fs/promises");
jest.mock("axios");

const mockConfig = `
input_text = "Hello world! Hello!"
find_word = "Hello"
replace_word = "Hi"
number_list = [1, 2, 3, 4, 5]
input_file = "input.txt"
output_file = "output.txt"
append_text = "Goodbye!"
api_url = "https://api.example.com/data"
`;

describe("GitHub Action", () => {
    const mockGetInput = jest.spyOn(core, "getInput");
    const mockSetOutput = jest.spyOn(core, "setOutput");
    const mockSetFailed = jest.spyOn(core, "setFailed");
    const mockWarning = jest.spyOn(core, "warning");
    const mockReadFile = fs.readFile as jest.MockedFunction<typeof fs.readFile>;
    const mockWriteFile = fs.writeFile as jest.MockedFunction<typeof fs.writeFile>;
    const mockAxiosGet = axios.get as jest.MockedFunction<typeof axios.get>;

    beforeEach(() => {
        jest.resetAllMocks();
        mockGetInput.mockReturnValue(".github/configs/setup-custom-action-by-ts.toml");
        mockReadFile.mockResolvedValue(mockConfig);
        mockAxiosGet.mockResolvedValue({ status: 200, data: {} });
    });

    it("should process text, count words, calculate sum and average correctly", async () => {
        await run();

        expect(mockSetOutput).toHaveBeenCalledWith("processed_text", "Hi world! Hi!");
        expect(mockSetOutput).toHaveBeenCalledWith("word_count", 3);
        expect(mockSetOutput).toHaveBeenCalledWith("sum", 15);
        expect(mockSetOutput).toHaveBeenCalledWith("average", 3);
    });

    it("should handle missing configuration file gracefully", async () => {
        mockReadFile.mockRejectedValue(new Error("File not found"));

        await run();

        expect(mockSetFailed).toHaveBeenCalledWith("Action failed with error: File not found");
    });

    it("should handle API request failures gracefully", async () => {
        mockAxiosGet.mockRejectedValue(new Error("API error"));

        await run();

        expect(mockWarning).toHaveBeenCalledWith("Failed to make API request: API error");
    });

    it("should check API reachability correctly", async () => {
        await run();

        expect(mockAxiosGet).toHaveBeenCalledWith("https://api.example.com/data", { timeout: 10000 });
    });

    it("should check API reachability correctly and warn on bad status", async () => {
        mockAxiosGet.mockResolvedValue({ status: 500, data: {} });

        await run();

        expect(mockWarning).toHaveBeenCalledWith("API is not reachable, status code: 500");
    });

    it("should read from the input file and append text to the output file correctly", async () => {
        // Mock for the first call (config file)
        mockReadFile.mockImplementationOnce(() => Promise.resolve(mockConfig));

        // Mock for the second call (input file)
        mockReadFile.mockImplementationOnce(() => Promise.resolve("Original file content."));

        const mockAppendText = "Goodbye!";
        await run();

        expect(mockReadFile).toHaveBeenCalledWith(".github/configs/setup-custom-action-by-ts.toml", "utf-8");
        expect(mockReadFile).toHaveBeenCalledWith("input.txt", "utf-8");
        expect(mockWriteFile).toHaveBeenCalledWith("output.txt", "Original file content.\nGoodbye!", {
            encoding: "utf-8",
        });
    });

    it("should handle file read errors gracefully", async () => {
        // First call to mock the config file
        mockReadFile.mockImplementationOnce(() => Promise.resolve(mockConfig));

        // Second call to mock the input file read error
        mockReadFile.mockImplementationOnce(() => Promise.reject(new Error("Failed to read input file")));

        await run();

        expect(mockWarning).toHaveBeenCalledWith("File operation failed: Failed to read input file");
    });

    it("should handle file write errors gracefully", async () => {
        // First call to mock the config file
        mockReadFile.mockImplementationOnce(() => Promise.resolve(mockConfig));

        // Second call to mock the input file read
        mockReadFile.mockImplementationOnce(() => Promise.resolve("Original file content."));

        // Mock write file to throw an error
        mockWriteFile.mockImplementationOnce(() => Promise.reject(new Error("Failed to write to output file")));

        await run();

        expect(mockWarning).toHaveBeenCalledWith("File operation failed: Failed to write to output file");
    });

    it("should use default values when config is empty", async () => {
        mockReadFile.mockResolvedValue("");

        await run();

        expect(mockSetOutput).toHaveBeenCalledWith("processed_text", "");
        expect(mockSetOutput).toHaveBeenCalledWith("word_count", 0);
        expect(mockSetOutput).toHaveBeenCalledWith("sum", 0);
        expect(mockSetOutput).toHaveBeenCalledWith("average", 0);
    });
});
