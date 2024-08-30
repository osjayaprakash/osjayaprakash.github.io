# frozen_string_literal: true

source 'https://rubygems.org'

require 'json'
require 'net/http'
versions = JSON.parse(Net::HTTP.get(URI('https://pages.github.com/versions.json')))

gem 'github-pages', versions['github-pages']

gem 'jekyll-seo-tag'

gem 'jekyll-sitemap', '~> 1.4'

gem 'jekyll-swiss', '~> 1.0'

gem 'jekyll-mentions', '~> 1.6'

# https://github.com/jekyll/jekyll/issues/8523
gem 'webrick', '~> 1.7'

group :development do
  gem 'colored'
  gem 'fuzzy_match'
  gem 'terminal-table'
end

group :test do
  gem 'html-proofer', '~> 3.0'
  gem 'rake'
  gem 'rspec'
  gem 'rubocop'
end

