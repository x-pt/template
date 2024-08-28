import { getInput, setFailed } from "@actions/core";
import { context, getOctokit } from "@actions/github";
import { run } from "../src";

jest.mock("@actions/core");
jest.mock("@actions/github");

describe("GitHub Action", () => {
    const mockGetInput = getInput as jest.MockedFunction<typeof getInput>;
    const mockSetFailed = setFailed as jest.MockedFunction<typeof setFailed>;
    const mockGetOctokit = getOctokit as jest.MockedFunction<typeof getOctokit>;

    const mockAddLabels = jest.fn();
    const mockOctokit = {
        rest: {
            issues: {
                addLabels: mockAddLabels,
            },
        },
    };

    beforeEach(() => {
        jest.clearAllMocks();
        mockGetInput.mockImplementation((name) => {
            switch (name) {
                case "gh-token":
                    return "gh-token-value";
                case "label":
                    return "label-value";
                default:
                    return "";
            }
        });
        mockGetOctokit.mockReturnValue(mockOctokit as any);
        (context as any).payload = { pull_request: { number: 1 } };
        (context as any).repo = { owner: "owner", repo: "repo" };
    });

    describe("run function", () => {
        it("should set failed if not run on a pull request", async () => {
            (context as any).payload = {};

            await run();

            expect(mockSetFailed).toHaveBeenCalledWith("This action can only be run on Pull Requests");
        });

        it("should add label to the pull request", async () => {
            await run();

            expect(mockGetInput).toHaveBeenCalledWith("gh-token", { required: true });
            expect(mockGetInput).toHaveBeenCalledWith("label", { required: true });
            expect(mockGetOctokit).toHaveBeenCalledWith("gh-token-value");
            expect(mockAddLabels).toHaveBeenCalledWith({
                owner: "owner",
                repo: "repo",
                issue_number: 1,
                labels: ["label-value"],
            });
            expect(mockSetFailed).not.toHaveBeenCalled();
        });

        it("should handle error and set failed", async () => {
            mockAddLabels.mockRejectedValueOnce(new Error("Test error"));

            await run();

            expect(mockAddLabels).toHaveBeenCalledWith({
                owner: "owner",
                repo: "repo",
                issue_number: 1,
                labels: ["label-value"],
            });
            expect(mockSetFailed).toHaveBeenCalledWith("Test error");
        });

        it("should set failed if gh-token is not provided", async () => {
            mockGetInput.mockImplementation((name, options) => {
                if (name === "gh-token" && options?.required) {
                    throw new Error("Input required and not supplied: gh-token");
                }
                return name === "label" ? "label-value" : "";
            });

            await run();

            expect(mockSetFailed).toHaveBeenCalledWith("Input required and not supplied: gh-token");
        });

        it("should set failed if label is not provided", async () => {
            mockGetInput.mockImplementation((name, options) => {
                if (name === "label" && options?.required) {
                    throw new Error("Input required and not supplied: label");
                }
                return name === "gh-token" ? "gh-token-value" : "";
            });

            await run();

            expect(mockSetFailed).toHaveBeenCalledWith("Input required and not supplied: label");
        });

        it("should set failed with generic message for unexpected errors", async () => {
            mockGetOctokit.mockImplementation(() => {
                throw new Error("Unexpected error");
            });

            await run();

            expect(mockSetFailed).toHaveBeenCalledWith("Unexpected error");
        });
    });
});
