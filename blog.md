---
layout: page
title: Blog
---

{% for post in site.posts %}
<div style="margin-bottom: 2rem; padding-bottom: 2rem; border-bottom: 1px solid #e8e8e8;">
  <div style="display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap;">
    <a href="{{ post.url }}" style="font-size:1.15rem; font-weight:600; color:#111827; text-decoration:none;">{{ post.title }}</a>
    <span style="font-size:0.85rem; color:#9ca3af; white-space:nowrap;">{{ post.date | date: "%-d %B %Y" }}</span>
  </div>
  {% if post.excerpt %}
  <p style="margin: 0.5rem 0 0; color:#4a5568; font-size:0.95rem; line-height:1.5;">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
  {% endif %}
</div>
{% endfor %}

{% if site.posts.size == 0 %}
<p style="color:#9ca3af;">No posts yet.</p>
{% endif %}
