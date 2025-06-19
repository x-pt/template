// Package cmd provides the command-line interface commands for the application.
package cmd

import (
	"fmt" // For formatted printing (e.g., version string)

	// Cobra is used for creating CLI applications.
	"github.com/spf13/cobra"
)

// versionCmd represents the 'version' command.
// When executed, it prints the version of the application.
var versionCmd = &cobra.Command{
	Use:   "version", // How the command is invoked (e.g., `my-app version`)
	Short: "Print the version number of {{cookiecutter.project_slug}}",
	Long:  `All software has versions. This is {{cookiecutter.project_slug}}'s version.`,
	// The Run function is executed when the 'version' command is called.
	Run: func(cmd *cobra.Command, args []string) {
		// The version is typically managed in one central place.
		// For this template, it's sourced directly from the cookiecutter.json value,
		// which is also often used to set the version in go.mod or via build flags.
		// In a real application, this might be a const string, a global variable
		// set at build time using ldflags (-X main.version=someversion), or read from a file.
		fmt.Printf("{{cookiecutter.project_slug}} version %s\n", "{{cookiecutter.project_version}}")
	},
}

// init is a special Go function called when the package is initialized.
// It's used here to add the versionCmd as a subcommand to the rootCmd.
// This makes the 'version' command available under the main application command.
func init() {
	rootCmd.AddCommand(versionCmd) // Add versionCmd to rootCmd so it's invokable.

	// Here you could define local flags for the version command if needed, for example:
	// versionCmd.Flags().BoolP("short", "s", false, "Print just the version number")
}
