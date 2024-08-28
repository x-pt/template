# Base image for building the application
FROM node:{{cookiecutter.node_version}}-bookworm AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json to the container
COPY package.json yarn.lock ./

# Install dependencies
RUN yarn install

# Copy the rest of the application code
COPY . .

# Build the application
RUN yarn build

# Separate stage for validation (build and test)
FROM builder AS validator

# Run tests as part of the validation
RUN yarn test

# Final image for running the application
FROM node:{{cookiecutter.node_version}}-slim-bookworm

LABEL author="{{cookiecutter.full_name}}"

# Set working directory
WORKDIR /app

# Copy built application code and node_modules from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package.json yarn.lock ./

# Expose the port the app runs on
EXPOSE 3000

# Define a health check
HEALTHCHECK --start-period=30s CMD curl -f http://localhost:3000 || exit 1

# Start the application
CMD ["node", "dist/index.js"]
