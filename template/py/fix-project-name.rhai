// get the project name
let project_name = variable::get("project-name");

// remove leading and trailing whitespace
project_name = project_name.trim();

// replace spaces with dashes
project_name = project_name.replace(" ", "-");
project_name = project_name.replace("_", "-");

// remove leading and trailing dashes and underscores
while project_name.starts_with("-") || project_name.starts_with("_") {
    project_name = project_name.sub(1, project_name.len());
}
while project_name.ends_with("-") || project_name.ends_with("_") {
    project_name = project_name.sub(0, project_name.len() - 1);
}

// check if the project name is empty
if project_name.len() == 0 {
    print("Error: Project name cannot be empty or consist solely of invalid characters.");
    throw "Invalid project name";
}

// set the project name
variable::set("project-name", project_name);
