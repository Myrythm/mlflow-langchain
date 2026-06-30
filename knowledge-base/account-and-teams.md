---
title: Account and Teams
category: account
---

# Account and Teams

This article explains how to manage your Nimbus workspace, member roles, team settings, and account preferences.

## Member Roles

Nimbus uses four roles to control what members can see and do:

| Role   | Capabilities                                                                                  |
|--------|-----------------------------------------------------------------------------------------------|
| Owner  | Full control over the workspace: billing, plan changes, member management, and all settings   |
| Admin  | Manage members and their roles, create and archive boards, configure integrations             |
| Member | Create and edit tasks on boards they have access to; cannot manage billing or members         |
| Guest  | View and comment on specific boards they are explicitly invited to; cannot access other boards|

**Guest access is available on Business and Enterprise plans only.** Guests do not count against your member seat count.

## Plan Member Limits

- **Free plan:** Up to 3 members (Owner + 2 others)
- **Pro, Business, and Enterprise:** Unlimited members

## Inviting Members

1. Navigate to **Settings → Members**.
2. Click **Invite members**.
3. Enter the email addresses of the people you want to invite (one per line or comma-separated).
4. Select a role from the dropdown: Owner, Admin, Member, or Guest.
5. Click **Send invites**.

Invitees receive an email with a join link. The link expires after 7 days. If it expires, resend the invitation from **Settings → Members → Pending invites**.

## Changing a Member's Role

1. Navigate to **Settings → Members**.
2. Find the member in the list.
3. Click the role badge next to their name and select a new role.
4. The change takes effect immediately.

Only Owners and Admins can change roles. An Owner cannot be downgraded by an Admin — only another Owner can change an Owner's role.

## Removing a Member

1. Navigate to **Settings → Members**.
2. Click the **...** menu next to the member's name.
3. Click **Remove from workspace** and confirm.

Removing a member revokes their access immediately. Their past activity (tasks created, comments) remains visible in boards.

## Transferring Ownership

Each workspace has exactly one Owner. To transfer ownership to another member:

1. Navigate to **Settings → Members**.
2. Find the member you want to promote.
3. Click **...** → **Transfer ownership**.
4. Confirm the transfer. Your role becomes Admin.

## Managing Your Profile

1. Click your avatar in the top-right corner and select **Profile settings**.
2. Update your name, avatar, and notification preferences.
3. Under **Security**, you can change your password or enable/disable TOTP-based 2FA.

## Workspace Settings

Workspace Owners and Admins can configure:

- **Workspace name and logo** — via **Settings → General**.
- **Default member role** for new invites — via **Settings → Members → Settings**.
- **2FA enforcement** — require all members to use 2FA (**Settings → Security**, available on Business and Enterprise).
- **SSO** — configure Google and Microsoft SSO (**Settings → Security → Single Sign-On**, Business and Enterprise); SAML SSO with SCIM is Enterprise-only.

## Deleting a Workspace

Only the Owner can delete a workspace. Deleting permanently removes all boards, tasks, and data.

1. Navigate to **Settings → General**.
2. Scroll to **Danger zone** and click **Delete workspace**.
3. Type the workspace name to confirm.
4. Click **Delete permanently**.

This action is irreversible. Export your data first via **Settings → General → Export data**.

## Support

For account-related help, reach out via the appropriate channel for your plan:

- **Free:** Community forum
- **Pro:** Priority email support
- **Business:** Priority support
- **Enterprise:** Dedicated customer success manager
