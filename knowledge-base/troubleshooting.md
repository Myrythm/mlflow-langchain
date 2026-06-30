---
title: Troubleshooting
category: troubleshooting
---

# Troubleshooting

This article covers common issues Nimbus users encounter and the steps to resolve them. If your issue is not listed here, please contact support using the channel for your plan.

## I Cannot Log In

**Issue:** You receive an "Invalid email or password" error.

1. Double-check your email address for typos.
2. Click **Forgot password** on the login page and follow the reset link sent to your email.
3. Check your spam or junk folder if the reset email does not arrive within a few minutes.
4. If your workspace uses SSO (Google, Microsoft, or SAML), use the **Sign in with SSO** option instead of entering a password directly.

**Issue:** You are redirected to your SSO provider but cannot complete sign-in.

- Ensure your SSO provider admin has assigned you to the Nimbus application.
- For SAML SSO issues, your workspace Owner or Admin can check the SSO configuration in **Settings → Security → SAML SSO**.

## I Am Not Receiving Email Notifications

1. Check your spam or junk folder.
2. Add `notifications@nimbus.io` to your email contacts or allowlist.
3. Verify your notification preferences: click your avatar → **Profile settings → Notifications** and confirm the relevant events are enabled.
4. If your organization uses email filtering (e.g., Microsoft Defender), ask your IT team to allowlist the Nimbus sending domain.

## An Integration Is Not Working

**Issue:** A connected integration stops sending notifications or events.

1. Navigate to **Settings → Integrations** and check whether the integration shows a yellow or red status indicator.
2. Click **Reconnect** to re-authorize the OAuth connection. OAuth tokens can expire or be revoked when you change your password on the provider side.
3. If the Slack integration is not posting notifications, verify that the Nimbus bot has been invited to the target Slack channel: in Slack, type `/invite @Nimbus` in the channel.
4. For GitHub/GitLab integrations, confirm that the repository has the Nimbus webhook enabled in the repository settings.

## Automation Rules Are Not Running

**Issue:** A rule you created is not triggering.

1. Open the board and click **Automate** to view the rule.
2. Confirm the rule is toggled **On**.
3. Check your workspace's automation run count for the current month: navigate to **Settings → Usage**. If you have reached your monthly limit, rules will not fire until the next billing cycle.
   - Free: 100 runs/month
   - Pro: 1,000 runs/month
   - Business: 10,000 runs/month
   - Enterprise: Custom/advanced limits
4. Review the **Run history** tab in the rule editor to see whether recent triggers were detected and what happened.

## File Upload Fails

**Issue:** An attachment fails to upload with a "file too large" error.

File upload limits per plan:

- **Pro:** 250 MB per file
- **Business and Enterprise:** 5 GB per file
- **Free:** File uploads are not included

If your file exceeds the limit for your plan, consider upgrading or compressing the file.

**Issue:** Upload fails intermittently.

1. Check your internet connection.
2. Try a different browser or disable browser extensions that intercept network requests.
3. Clear your browser cache and reload Nimbus.

## I Cannot Add More Members

**Issue:** An error appears when inviting new members.

- **Free plan:** You are limited to 3 members total. To add more, upgrade to Pro ($12/member/month billed annually or $14/member/month billed monthly) or higher.
- **Pending invites:** Pending (not yet accepted) invites count toward your member total on the Free plan. Rescind unused invites from **Settings → Members → Pending invites**.

## A Board or Task Is Missing

1. Check the **Archived** section: go to **Boards** in the sidebar, then click **View archived boards**.
2. Use the global search (press `/`) to search for the task by name.
3. Check whether a team member with Admin or Owner permissions may have archived or deleted it from the board activity log (click **Activity** in the board header).

## Performance Is Slow

1. Clear your browser cache and hard-reload the page (Ctrl+Shift+R on Windows/Linux, Cmd+Shift+R on macOS).
2. Check the [Nimbus status page](https://status.nimbus.io) for any active incidents or degraded services.
3. Disable browser extensions one by one to rule out interference.
4. If you are on a large board with many tasks, try filtering by assignee or date to reduce the visible task count.

## Contact Support

If none of the steps above resolve your issue:

- **Free plan:** Post in the community forum at community.nimbus.io.
- **Pro plan:** Email our priority support team from **Settings → Help → Contact support**.
- **Business plan:** Contact priority support via **Settings → Help → Contact support**.
- **Enterprise plan:** Reach out to your dedicated customer success manager directly.

When contacting support, include your workspace URL, a description of the issue, and any error messages or screenshots.
