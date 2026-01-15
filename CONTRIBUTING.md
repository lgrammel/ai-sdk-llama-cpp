# Contributing

Thank you for your interest in contributing to ai-sdk-llama-cpp!

## Development Setup

See [AGENTS.md](./AGENTS.md) for detailed setup instructions, project structure, and development commands.

## Making Changes

1. Fork the repository and create a new branch for your changes
2. Make your changes and ensure tests pass: `pnpm test:run`
3. Run the formatter: `pnpm format:fix`

## Changesets

This project uses [changesets](https://github.com/changesets/changesets) for versioning and changelog management.

### Adding a Changeset

When you make changes that should be released, add a changeset:

```bash
pnpm changeset
```

This will prompt you to:
1. Select the `ai-sdk-llama-cpp` package
2. Choose the semver bump type:
   - **patch**: Bug fixes, documentation updates
   - **minor**: New features, non-breaking changes
   - **major**: Breaking changes
3. Write a summary of your changes (this appears in the changelog)

The changeset file will be created in the `.changeset` directory. Include this file in your pull request.

### When to Add a Changeset

Add a changeset when your changes:
- Fix a bug
- Add a new feature
- Change existing behavior
- Update dependencies in a meaningful way

You don't need a changeset for:
- Documentation-only changes (unless they fix incorrect docs)
- Changes to tests, examples, or dev tooling
- Refactoring with no user-facing changes

## Releasing (Maintainers Only)

Releases are done manually by maintainers.

### 1. Version Packages

Run the version command to consume all changesets and update package versions:

```bash
pnpm changeset:version
```

This will:
- Update `package.json` versions
- Update `CHANGELOG.md` files
- Remove the consumed changeset files

Review the changes and commit them:

```bash
git add .
git commit -m "chore: version packages"
git push
```

### 2. Publish to npm

Ensure you're logged in to npm:

```bash
npm login
```

Then publish:

```bash
pnpm changeset:publish
```

This will build the TypeScript and publish the package to npm.

### 3. Create a Git Tag

After publishing, create and push a git tag:

```bash
git tag v$(node -p "require('./packages/ai-sdk-llama-cpp/package.json').version")
git push --tags
```

### 4. Create a GitHub Release

Optionally, create a GitHub release from the tag with the changelog contents.
