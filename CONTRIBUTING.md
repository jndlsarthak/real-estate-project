# Contributing Guidelines

## Workflow

1. **Create an Issue**  
   Describe the task, improvement, or stage you're working on.

2. **Create a Branch**  
   Use the format: `stageX-taskname` (e.g., `stage2-ml-design`).

3. **Do Your Work**  
   Make changes and commit regularly with clear messages.

4. **Push to GitHub**  
   Push your branch using:  
   `git push origin stageX-taskname`

5. **Open a Pull Request**  
   Target the `main` branch. Provide a brief summary of your changes.

6. **Request Review and Merge**  
   Once reviewed and approved, merge the PR. Delete the branch after merge.

## Branch Naming

- Format: `stageX-taskname`  
  Examples:
  - `stage1-eda`
  - `stage2-ml-design`
  - `stage3-model-training`

## Commit Messages

- Use meaningful messages (e.g., `Fix room_count parsing`, `Add correlation heatmap`).

## Code & Files

- Keep notebooks in the `/notebook` folder.
- Add processed data to `/data` only if needed.
- Reports go in `/reports`.

