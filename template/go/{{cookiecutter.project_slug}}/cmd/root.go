package cmd

import (
	"fmt"
	"github.com/spf13/viper"
	"os"

	"github.com/spf13/cobra"
)

var cfgFile string

var rootCmd = &cobra.Command{
	Use:   "{{cookiecutter.project_slug}}",
	Short: "{{cookiecutter.project_desc}}",
	Long: `A long desc for the {{cookiecutter.project_slug}}.
Such as:
- Feature1
- Feature2
- Feature3
`,
}

func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag.
		viper.SetConfigFile(cfgFile)
	} else {
		viper.AddConfigPath("$HOME/.config/blog-go")
		viper.AddConfigPath("./config/")

		viper.SetConfigName("config")
		viper.SetConfigType("yaml")
	}

	// set the default values
	setDefaultConfig()

	viper.AutomaticEnv() // read in environment variables that match

	// If a config file is found, read it in.
	if err := viper.ReadInConfig(); err != nil {
		fmt.Printf("%s\n\n\n", err.Error())
		_ = rootCmd.Help()
		os.Exit(1)
	}
	fmt.Printf("======== Using config file: %s ========\n", viper.ConfigFileUsed())
}

func setDefaultConfig() {
	// Some default values
}
