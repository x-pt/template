.PHONY: dev

dev:
	# create a image "{{project-name}}" with tag "latest"
	# based on the Dockerfile in the current directory
	docker image build . -t {{project-name}}
	docker container run --rm {{project-name}}
