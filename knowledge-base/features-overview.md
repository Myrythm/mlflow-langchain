---
title: Features Overview
category: features
---

# Features Overview

Nimbus is a cloud-based project-management platform built around four core capabilities: boards, timeline view, automation, and reporting. This article describes each feature and which plans include it.

## Boards

Boards are the foundation of Nimbus. Every project lives on a board, organized as columns and task cards.

- **Drag-and-drop tasks** between columns to update their status.
- **Task cards** hold a title, description, assignees, due date, labels, attachments, and comments.
- **Subtasks** let you break large pieces of work into smaller steps.
- **Board views** include Kanban (default), List, and Calendar.
- **Multiple boards per workspace:** Free plan supports 2 boards; Pro, Business, and Enterprise support unlimited boards.

## Timeline View

The Timeline view gives you a Gantt-style visualization of tasks and their durations across a calendar.

- Available on **Pro, Business, and Enterprise** plans.
- Drag tasks along the timeline to reschedule them.
- Draw dependencies between tasks to model blockers and sequencing.
- Zoom between day, week, and month views.

To open the Timeline view, click the **Timeline** tab at the top of any board.

## Automation

Automation rules run actions automatically when a trigger condition is met — for example, moving a task to "Done" when all subtasks are complete, or notifying a Slack channel when a task is overdue.

**Automation runs per month by plan:**

| Plan       | Automation Runs/Month |
|------------|-----------------------|
| Free       | 100                   |
| Pro        | 1,000                 |
| Business   | 10,000                |
| Enterprise | Custom / advanced     |

**Creating an automation rule:**

1. Open a board and click the **Automate** button in the top toolbar.
2. Click **+ New rule**.
3. Select a **Trigger** (e.g., "Task status changes to Done").
4. Add one or more **Actions** (e.g., "Assign to member", "Send Slack notification", "Move to board").
5. Click **Save rule**.

Rules run in the order they are listed. You can pause, edit, or delete rules at any time.

## Reporting Dashboards

Reporting dashboards provide aggregate views of work across your workspace — task completion rates, overdue counts, workload by member, and more.

- Available on **Business and Enterprise** plans.
- Navigate to **Reports** in the left sidebar to access dashboards.
- Dashboards update in real time as tasks change.
- Export reports as CSV or PDF from the **Export** menu.

## File Uploads

Attach files directly to task cards. Supported formats include images, PDFs, documents, and archives.

| Plan       | File Upload Limit |
|------------|-------------------|
| Free       | —                 |
| Pro        | 250 MB per file   |
| Business   | 5 GB per file     |
| Enterprise | Custom            |

## Integrations

Nimbus integrates with Slack, GitHub, GitLab, Google Drive, Figma, Zapier, and outgoing webhooks. All integrations are accessible on Pro, Business, and Enterprise plans. Connect via **Settings → Integrations** using OAuth.

See the [Integrations](integrations.md) article for setup instructions.

## Guest Access

Invite clients, contractors, or external stakeholders as Guests to specific boards without giving them access to the full workspace. Guest access is available on **Business and Enterprise** plans.

## Search and Filters

- Use the global **Search** bar (keyboard shortcut: `/`) to find tasks, boards, or members by keyword.
- Apply **Filters** on any board to show only tasks matching criteria such as assignee, label, due date range, or custom fields.
- Save frequently used filters as **Saved views** for quick access.

## Keyboard Shortcuts

Press `?` anywhere in Nimbus to open the keyboard shortcuts reference panel. Common shortcuts include:

- `N` — Create a new task
- `/` — Open global search
- `T` — Switch to Timeline view
- `G then B` — Go to Boards home
