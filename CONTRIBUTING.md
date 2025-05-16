# OSML Contributing Guidelines

Thank you for your interest in contributing to our project! This document will guide you through the process of
contributing to our repository using a Trunk-Based Development workflow. Please read the guidelines carefully to
submit your pull requests effectively.

## Table of Contents

- [Trunk-Based Development](#trunk-based-development)
- [Linting](#linting)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Issue Tracking](#issue-tracking)

## Trunk-Based Development

![Trunk-Based Development](https://github.com/awslabs/osml-imagery-toolkit/assets/4109909/0b3c03ae-4518-471e-9331-da850f0d2e22)

We follow a trunk-based development model to manage our codebase.
This guide provides a step-by-step example of implementing a Trunk-Based Development (TBD) workflow for development.
The branches developers should create directly off of `main` (our trunk) as part of this workflow are:

- `feature/*`: `feature` branches are created for the development of new features or significant enhancements.

### 1. Creating Feature (Internal Developer)

#### Step 1: Clone the Repository
Clone the repository to your local machine.
```bash
git clone git@github.com:aws-solutions-library-samples/osml-imagery-toolkit.git
```

#### Step 2: Create a New Branch
Create a new branch off `main`, naming it related to the work being done, in this case a feature.
```bash
git checkout -b feature/<feature_id>
```

#### Step 3: Develop and Commit Changes
On the new branch, code, and add commits as necessary.

#### Step 4: Push the New Branch
Push the new branch to trigger unit tests, static code analysis, Sonar cube checks, security checks, etc.,
before merging to `main`.
```bash
git push origin feature/<feature_id>
```

#### Step 5: Create a Pull Request (PR)
Open a PR against `main` to kick off a discussion.
Open a PR as soon as possible, even if it's not ready (mark it as WIP).
This ensures ample time for review and discussion, improving code quality.

#### Step 6: PR Review, Rebase, and Merge
After your PR is reviewed, make any necessary changes by repeating steps 3 and 4.
A team member will merge your PR after approval,
ensuring it's reviewed by someone who has not contributed to the branch.
Also ensure that your branch is rebased against the latest changes to `main`.
```bash
git pull --rebase origin main
git push origin feature/<feature_id>
```
Now you are ready to merge your changes!

### 2. Contributing via Fork and Pull Request (External Developer)

We welcome contributions from the community!
If you're looking to contribute to our project, please follow these steps to fork the repository,
make your changes, and submit a pull request (PR) for review.

#### Step 1: Fork the Repository

1. Navigate to the GitHub page of our repository.
2. In the top-right corner of the page, click the **Fork** button. This creates a copy of the repository in your GitHub account.

#### Step 2: Clone Your Fork

1. On your GitHub page for the forked repository, click the **Clone** or **Download** button and copy the URL.
2. Open your terminal or command prompt.
3. Clone your fork using the command:
   ```bash
   git clone [URL of the forked repository]
   ```
4. Navigate into the cloned repository:
   ```bash
   cd [repository-name]
   ```

#### Step 3: Create a New Branch

Create a new branch for your changes based off the latest `main`:
   ```bash
   git checkout -b [branch-name]
   ```
   Follow the branch naming conventions mentioned earlier.

If you wish to work with a stable release version of our project,
you might want to clone a specific release tag rather than the latest state of the main branch.
Here's how you can do that:

1. **List Available Tags**: First, to see the available tags (release versions), navigate to the repository on GitHub, then go to the **Tags** section in the **Releases** tab. Alternatively, you can list tags from the command line using:
   ```bash
   git ls-remote --tags [URL of the original repository]
   ```
2. **Clone the Repository**: If you haven't already, clone the repository using:
   ```bash
   git clone [URL of the original repository]
   ```
3. **Checkout the Tag**: Navigate into the cloned repository directory:
   ```bash
   cd [repository-name]
   ```
   Then, checkout the specific tag you're interested in working with:
   ```bash
   git checkout tags/[tag-name]
   ```
   Replace `[tag-name]` with the desired tag.
   This will put you in a 'detached HEAD' state, which is fine for browsing a tag.

If you plan to make changes starting from this tag and contribute back,
it's a good idea to create a new branch from this tag:

```bash
git checkout -b [new-branch-name] tags/[tag-name]
```

This way, you can start your changes from a specific version of the project.

#### Step 4: Make Your Changes

- Make the necessary changes in your branch. Feel free to add, edit, or remove content as needed.

#### Step 5: Commit Your Changes

1. Stage your changes for commit:
   ```bash
   git add .
   ```
2. Commit your changes with a meaningful commit message:
   ```bash
   git commit -m "A descriptive message explaining your changes"
   ```

#### Step 6: Push Your Changes

1. Push your changes to your fork:
   ```bash
   git push origin [branch-name]
   ```

#### Step 7: Create a Pull Request

1. Navigate to your forked repository on GitHub.
2. Click the **Pull Request** button.
3. Ensure the base repository is the original repository you forked from and the base branch is the one you want your changes pulled into.
4. Select your branch as the compare branch.
5. Fill in a title and description for your pull request explaining the changes you've made.
6. Click **Create Pull Request**.

#### Final Step: Await Review

- Once your pull request is submitted, our team will review your changes. We may request further modifications or provide feedback before merging your changes.
- Keep an eye on your GitHub notifications for comments or requests for changes from the project maintainers.

## Linting

This package uses a number of tools to enforce formatting, linting, and general best practices:
* [eslint](https://github.com/pre-commit/mirrors-eslint) to check pep8 compliance and logical errors in code
* [prettier](https://github.com/pre-commit/mirrors-prettier) to check pep8 compliance and logical errors in code
* [pre-commit](https://github.com/pre-commit/pre-commit-hooks) to install and control linters in githooks

After pulling the repository, you should enable auto linting git commit hooks by running:

```bash
python3 -m pip install pre-commit
pre-commit install
```

In addition to running the linters against files included in a commit, you can perform linting on all the files
in the package by running:
```bash
pre-commit run --all-files --show-diff-on-failure
```
or if using tox
```bash
tox -e lint
````

## Code Style

We maintain a consistent code style throughout the project, so please ensure your changes align with our existing style.
Take a look at the existing codebase to understand the patterns and conventions we follow.

## Commit Messages

The [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) specification is a
lightweight convention on top of commit messages. It provides an easy set of rules for creating an explicit
commit history, which makes it easier to write automated tools on top of. This convention dovetails with SemVer
by describing the features, fixes, and breaking changes made in commit messages. Please refer to the linked
documentation for examples and additional context.

<code>&lt;type&gt;[optional scope]: &lt;description&gt;

[optional body]

[optional footer(s)]
</code>

## Issue Tracking

We use the issue tracking functionality of GitHub to manage our project's roadmap and track bugs or feature requests.
If you encounter any problems or have a new idea, please search the issues to ensure it hasn't already been reported.
If necessary, open a new issue providing a clear description, steps to reproduce, and any relevant information.

We greatly appreciate your effort and contribution to our project! Let's build something awesome together!
