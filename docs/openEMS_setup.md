```
brew update
brew upgrade
brew cleanup
```

```
brew tap thliebig/openems https://github.com/thliebig/openEMS-Project.git
```

```
brew install --HEAD openems
```

```
cd ~/Library/Caches/Homebrew/openems--git/CSXCAD/python
python setup.py build_ext -I /opt/homebrew/opt/openems/include -L /opt/homebrew/opt/openems/lib -R /opt/homebrew/opt/openems/lib
```

```
cd ~/Library/Caches/Homebrew/openems--git/openEMS/python
python setup.py build_ext -I /opt/homebrew/opt/openems/include -L /opt/homebrew/opt/openems/lib -R /opt/homebrew/opt/openems/lib
```