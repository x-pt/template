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
    const mockInfo = jest.spyOn(core, "info");
    const mockReadFile = fs.readFile as jest.MockedFunction<typeof fs.readFile>;
    const mockWriteFile = fs.writeFile as jest.MockedFunction<typeof fs.writeFile>;
    const mockAxiosGet = axios.get as jest.MockedFunction<typeof axios.get>;

    beforeAll(() => {
        mockGetInput.mockReturnValue(".github/configs/setup-custom-action-by-ts.toml");
    });

    beforeEach(() => {
        jest.resetAllMocks();
        setupDefaultMocks();
    });

    const setupDefaultMocks = () => {
        mockReadFile.mockResolvedValue(mockConfig);
        mockAxiosGet.mockResolvedValue({ status: 200, data: {} });
    };

    const setupFileMocks = (inputContent: string) => {
        mockReadFile.mockImplementation((path) => {
            if (path === ".github/configs/setup-custom-action-by-ts.toml") {
                return Promise.resolve(mockConfig);
            }
            if (path === "input.txt") {
                return Promise.resolve(inputContent);
            }
            return Promise.reject(new Error("Unexpected file read"));
        });
    };

    describe("Text Processing", () => {
        it("should process text, count words, calculate sum and average correctly", async () => {
            await run();

            expect(mockSetOutput.mock.calls).toMatchSnapshot();
        });

        it("should handle empty input text", async () => {
            mockReadFile.mockResolvedValue(`
input_text = ""
number_list = []
`);

            await run();

            expect(mockSetOutput.mock.calls).toMatchSnapshot();
        });
    });

    describe("File Operations", () => {
        it("should read from the input file and append text to the output file correctly", async () => {
            setupFileMocks("Original file content.");

            await run();

            expect(mockReadFile).toHaveBeenCalledWith("input.txt", "utf-8");
            expect(mockWriteFile).toHaveBeenCalledWith("output.txt", "Original file content.\nGoodbye!", {
                encoding: "utf-8",
            });
            expect(mockInfo).toHaveBeenCalledWith("Appended text to file: output.txt");
        });

        it("should handle missing input file gracefully", async () => {
            mockReadFile.mockRejectedValue(new Error("File not found"));

            await run();

            expect(mockSetFailed).toHaveBeenCalledWith("Action failed with error: File not found");
        });

        it("should handle file write errors gracefully", async () => {
            setupFileMocks("Original file content.");
            mockWriteFile.mockRejectedValue(new Error("Failed to write to output file"));

            await run();

            expect(mockSetFailed).toHaveBeenCalledWith(
                "Action failed with error: File operation failed: Failed to write to output file",
            );
        });
    });

    describe("API Operations", () => {
        it("should check API reachability correctly", async () => {
            await run();

            expect(mockAxiosGet).toHaveBeenCalledWith("https://api.example.com/data", { timeout: 10000 });
            expect(mockInfo).toHaveBeenCalledWith("API https://api.example.com/data is reachable.");
        });

        it("should handle API request failures gracefully", async () => {
            mockAxiosGet.mockRejectedValue(new Error("API error"));

            await run();

            expect(mockWarning).toHaveBeenCalledWith("Failed to make API request: API error");
        });

        it("should warn on bad API status", async () => {
            mockAxiosGet.mockResolvedValue({ status: 500, data: {} });

            await run();

            expect(mockWarning).toHaveBeenCalledWith("API is not reachable, status code: 500");
        });
    });

    describe("Configuration Handling", () => {
        it("should use default values when config is empty", async () => {
            mockReadFile.mockResolvedValue("");

            await run();

            expect(mockSetOutput.mock.calls).toMatchSnapshot();
        });

        it("should handle missing configuration file gracefully", async () => {
            mockReadFile.mockRejectedValue(new Error("Config file not found"));

            await run();

            expect(mockSetFailed).toHaveBeenCalledWith("Action failed with error: Config file not found");
        });
    });
});
