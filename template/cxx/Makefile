.PHONY: dev clean

# create a image "{{project-name}}" with tag "latest"
# based on the Dockerfile in the current directory
dev:
	docker image build . -t {{project-name}}
	docker container run --rm {{project-name}}

clean:
	docker image rm {{project-name}}
