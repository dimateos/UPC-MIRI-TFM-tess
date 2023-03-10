
# Run before collecting the tests actually allows to modify the env (other open interactive windows may block it tho)
def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    print("pytest_configure")
    if True:
        print("\n>> TESTS CONFIGURE")

        # check correct python env!
        import os, sys
        print("> sys.version_info", sys.version_info)
        print("> sys.executable", sys.executable)
        print("> os.getcwd()", os.getcwd())

        # redo the setup, will detect if the current version is already the last one
        try:
            os.system(sys.executable + " setup.py develop")
            # os.system(sys.executable + " -m pip install --editable .") # interestingly pip wont work
        except:
            print("TESTS FAILED CYTHON RECOMPILE...")
