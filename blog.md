---
layout: blog
title: 'NOTES'
---

<!-- * [pinned] [notes on investment or finance](/notes-on-finance-or-investment.html)
* [pinned] [ai papers]() -->

<ul class="post-list">
  {%- for post in posts -%}
  <li class="post-item">
    <div class="post-meta">
      <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
      <span class="separator">·</span>
      <span class="read-time">{{ post.content | number_of_words | divided_by: 200 }} min read</span>
    </div>
    <a class="post-link" href="{{ post.url | relative_url }}">
      {{ post.title | escape }}
    </a>
    {%- if post.excerpt -%}
      <p class="post-excerpt">{{ post.excerpt | strip_html | truncate: 160 }}</p>
    {%- endif -%}
  </li>
  {%- endfor -%}
</ul>
