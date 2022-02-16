#!/bin/sh
echo trappedClassic.lay
python3 pacman.py -l trappedClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo contestClassic.lay
python3 pacman.py -l contestClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo powerClassic.lay
python3 pacman.py -l powerClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo capsuleClassic.lay
python3 pacman.py -l capsuleClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo mediumClassic.lay
python3 pacman.py -l mediumClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo smallClassic.lay
python3 pacman.py -l smallClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo minimaxClassic.lay
python3 pacman.py -l minimaxClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo testClassic.lay
python3 pacman.py -l testClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo trickyClassic.lay
python3 pacman.py -l trickyClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo originalClassic.lay
python3 pacman.py -l originalClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo customClassic.lay
python3 pacman.py -l customClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
echo openClassic.lay
python3 pacman.py -l openClassic.lay -p ExpectimaxAgent -a evalFn=better -q -n 5
python3 autograder.py -q test_evaluation_fn
