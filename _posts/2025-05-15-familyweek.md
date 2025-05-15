---
layout: post
title: Family Week - Building a Web App For a Weekly Family Life Journey
---

![family week]({{ site.baseurl }}/images/familyweek.png)

Inspired by the [Life in Weeks project by Gina Trapani](https://github.com/ginatrapani), I set out to build a customized version tailored for our family. The result is *Family Week*, a lightweight web app designed to document and reflect on our shared moments—structured weekly, but flexible in content.

Based on the original idea, I implemented a few enhancements to better serve our needs:

- **Image support**: Upload up to 10 images per week, with full-screen viewing.
- **Video integration**: Embed a video per entry with automatic thumbnail generation (with fallbacks for browser compatibility).
- **Audio support**: We frequently use voice memos, so native audio playback is supported for each event.
- **Improved interactivity**: Clicking an event opens a persistent pop-up that stays active until another event is hovered over.
- **Access control**: The app contains personal data, so access is restricted to family members only.
- **Family Tree**: A simple static family tree for our daughter to explore and understand her roots.

## Challenges

### Managing Sensitive Data

One of the primary concerns was safeguarding privacy. I’ve configured the app to be non-indexable and gated via [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/). This allows email-based authentication with One-Time-Password and session control.

### Cost-Efficient Hosting

To keep the project lean and mostly free:

- The domain was registered via [Porkbun](https://porkbun.com/), which offers affordable rates.
- The site is hosted as a private GitHub project and served statically.
- For media optimization, I wrote a Python utility that:
  - Uses [tinify](https://tinypng.com/) for image compression.
  - Leverages [HandBrake](https://handbrake.fr/downloads.php) for video and audio compression.

These steps help reduce bandwidth and improve performance without relying on heavy backend infrastructure.

## Technical Stack

**Core Framework:**
- **Hugo** — A fast static site generator used to build and deploy the app

**Frontend Technologies:**
- **HTML/CSS** — Standard web markup and styling
- **SCSS/Sass** — For modular and maintainable stylesheets
- **JavaScript** — Interactive client-side logic
- **jQuery** — Simplifies DOM operations and event handling
- **Bootstrap** — UI components and responsive layout

**Extras:**
- **No Database** - All images are stored in the static folder and content is added via Markdown files as per standard with Hugo. 

The result is a static, performant, and secure family journaling tool that’s highly tailored to our routines without relying on large SaaS platforms or monthly hosting plans.
