import { getInput, setFailed } from "@actions/core";
import { context, getOctokit } from "@actions/github";

export async function run(): Promise<void> {
    try {
        const token = getInput("gh-token", { required: true });
        const label = getInput("label", { required: true });

        const pullRequest = context.payload.pull_request;
        if (!pullRequest) {
            setFailed("This action can only be run on Pull Requests");
            return;
        }

        const octokit = getOctokit(token);
        await octokit.rest.issues.addLabels({
            ...context.repo,
            issue_number: pullRequest.number,
            labels: [label],
        });
    } catch (error) {
        setFailed(error instanceof Error ? error.message : "An unexpected error occurred");
    }
}

if (!process.env.JEST_WORKER_ID) {
    run().catch((error) => {
        console.error("Unhandled error:", error);
        process.exit(1);
    });
}
