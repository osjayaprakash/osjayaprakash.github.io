
* Setup in local machine 
	* Clone the repo, install ruby and 
	```
	gh repo clone osjayaprakash/osjayaprakash.github.io
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
	bundle install
	```
	* [optional] with newer ruby missing some lib
	```
	bundle add webrick
	```
	* run your jekyll site locally
	```

	```