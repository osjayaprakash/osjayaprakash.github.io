* Clone the repo,  and 
```
gh repo clone osjayaprakash/osjayaprakash.github.io
```
* install ruby
```
brew install ruby
```
* Set PATH for ruby bin and lib
```
echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc
echo 'export LDFLAGS="-L/opt/homebrew/opt/ruby/lib"' >> ~/.zshrc
echo 'export CPPFLAGS="-I/opt/homebrew/opt/ruby/include"' >> ~/.zshrc
```
* bundle install
```
bundle init
```
* [optional] with newer ruby missing some lib
```
bundle add webrick
bundle update github-pages
bundle install
```
* run your jekyll site locally
```
bundle exec jekyll serve --incremental
```