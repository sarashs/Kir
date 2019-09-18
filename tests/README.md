Having the `__init__.py` in here helps with the auto discovery of `pytest`.  
Once it finds `test_*.py` goes up through the directories until it reaches
one that doesn't have an `__init__.py`.
That becomes the 'basedir'.
Then it does `sys.path.insert(0, basedir)` and uses that to import the test
modules.  
This has the side effect of making the package code available as well.  
https://docs.pytest.org/en/latest/goodpractices.html#test-discovery
