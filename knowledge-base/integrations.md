---
title: Integrations
category: integrations
---

# Integrations

Nimbus connects with the tools your team already uses. All integrations are available on the Pro, Business, and Enterprise plans. Most integrations are configured via **Settings → Integrations** using OAuth; Zapier uses an API key instead.

## Available Integrations

| Integration   | What it does                                                              |
|---------------|---------------------------------------------------------------------------|
| Slack         | Receive task notifications and create tasks directly from Slack messages  |
| GitHub        | Link pull requests and issues to Nimbus tasks; sync status automatically  |
| GitLab        | Link merge requests and issues; view code activity alongside tasks        |
| Google Drive  | Attach Drive files to tasks; browse and search Drive without leaving Nimbus|
| Figma         | Embed Figma frames and prototypes directly on task cards                  |
| Zapier        | Connect Nimbus to thousands of apps via automated Zaps                    |
| Outgoing webhooks | Send real-time event payloads to any URL when tasks or boards change  |

## Connecting an Integration

1. Navigate to **Settings → Integrations**.
2. Find the integration you want to enable and click **Connect**.
3. You are redirected to the provider's OAuth authorization page.
4. Sign in to the provider account you want to link and click **Authorize**.
5. You are returned to Nimbus. The integration card shows a green **Connected** status.

For outgoing webhooks, you provide the destination URL instead of going through OAuth.

## Slack Integration

Once connected, you can:

- Choose which boards and events trigger Slack notifications (task created, status changed, comment added, due date approaching).
- Map a Nimbus board to a Slack channel from **Settings → Integrations → Slack → Configure**.
- Create Nimbus tasks from any Slack message using the **Create task** shortcut in the message action menu.

## GitHub and GitLab Integrations

1. Connect the integration via **Settings → Integrations** using OAuth.
2. On any Nimbus task, paste a GitHub pull request URL or GitLab merge request URL into the task description or the **Link** field. Nimbus automatically fetches the title and current status.
3. When the PR/MR is merged or closed, the linked Nimbus task is optionally moved to your configured "Done" column automatically.

## Google Drive Integration

1. After connecting, open any task and click the **Attach** button.
2. Select **Google Drive** from the file picker.
3. Browse or search your Drive and select files to attach. Files are linked — not copied — so they always show the latest version.

## Figma Integration

1. Copy a Figma frame or prototype link from Figma.
2. Paste it into a Nimbus task description. Nimbus renders an inline preview automatically.

## Zapier Integration

1. Connect the Nimbus app in your Zapier account using your Nimbus API key (found in **Settings → API**).
2. Use Nimbus as a Zap trigger (e.g., "New task created") or action (e.g., "Create task in Nimbus").

## Outgoing Webhooks

1. Navigate to **Settings → Integrations → Webhooks** and click **Add webhook**.
2. Enter the destination URL that should receive the payloads.
3. Select the events to subscribe to (task created, task updated, comment added, etc.).
4. Click **Save**. Nimbus sends a test ping to verify the endpoint is reachable.

## Disconnecting an Integration

1. Navigate to **Settings → Integrations**.
2. Click the integration you want to remove.
3. Click **Disconnect** and confirm. Nimbus revokes its OAuth token and stops sending events.

## Plan Availability

All integrations require a **Pro, Business, or Enterprise** plan. Free plan workspaces do not have access to integrations.
