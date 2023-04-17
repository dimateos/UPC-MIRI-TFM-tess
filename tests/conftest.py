
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

        def _recompile_module():
            # redo the setup, will detect if the current version is already the last one
            ret = os.system(sys.executable + " setup.py develop")

            # # interestingly pip wont work
            # ret = os.system(sys.executable + " -m pip install --editable .")
            return ret

        try:
            print("> recompiling module...")
            # compilation may fail as explained in the error except message
            if _recompile_module(): raise Exception("os level error")

        except:
            try:
                # try to delete manually (usually will fail too)
                print("> deleting .pyd and recompiling module...")
                ret = os.system("cd tess && del /Q /F _voro.*.pyd")
                if ret: raise Exception("os level error")

                if _recompile_module(): raise Exception("os level error")

            except:
                # just print a message
                print("""
                -----------------------------------------------------------------------------------
                                    *** TESTS FAILED CYTHON RECOMPILE... ***
                Probably because some python interpreter instance is accessing the module
                * OTHERWISE, consider deleting .pyd manually to free the module
                * Other issues will require reinstalling the module, e.g. the test collection requires it installed...
                -----------------------------------------------------------------------------------
                """)
