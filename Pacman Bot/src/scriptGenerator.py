import os, sys

# Run "chmod +x testScript.sh" on terminal to run the testing script

def makescript():
    n = 5
    if len(sys.argv) > 1:
        n = sys.argv[1]
    os.chdir("layouts")
    layouts = os.listdir()
    os.chdir("..")
    f = open("testScript.sh", "w")
    f.write("#!/bin/sh\n")
    for layout in layouts:
        f.write("echo " + layout + "\n")
        f.write("python3 pacman.py -l " + layout + " -p ExpectimaxAgent -a evalFn=better -q -n " + str(n) + "\n")
    f.write("python3 autograder.py -q test_evaluation_fn\n")
    f.close()

makescript()
