#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/


from unittest import TestCase as BaseTestCase, TestLoader, TestSuite, TestResult
import numpy
import sys
import numpy as np


from essentia_rust import EssentiaException

# any and all are only defined in python >= 2.5
if int(sys.version.split()[0].split('.')[1]) < 5:
    def all(l):
        for elem in l:
            if not elem:
                return False
        return True


    def any(l):
        for elem in l:
            if elem:
                return True
        return False


def filedir():
    '''This returns the directory that the calling file is in.'''
    callingFile = sys._getframe(1).f_globals['__file__']
    from os.path import dirname
    return dirname(callingFile)


def readValue(filename):
    return float(open(filename).read())


def readVector(filename):
    return [float(x) for x in open(filename).read().strip().split()]


def readVectorTwoColumns(filename):
    vector = []
    for ii, x in enumerate(open(filename).read().strip().split()):
        if ii % 2 == 1:
            vector.append(float(x))
    return vector


def readComplexVector(filename):
    values = open(filename).read().strip().split()
    values = [v.replace('(', '') for v in values]
    values = [v.replace(')', 'j') for v in values]
    values = [v.replace(',-', '-') for v in values]
    values = [v.replace(',', '+') for v in values]
    return [complex(v) for v in values]


def readMatrix(filename):
    return np.array([[float(value) for value in line.strip().split()] for line in open(filename).readlines()])


def allTests(testClass):
    return TestLoader().loadTestsFromTestCase(testClass)


class TestCase(BaseTestCase):

    def assertValidNumber(self, x):
        self.assert_(not numpy.isnan(x))
        self.assert_(not numpy.isinf(x))

    def assertValidPool(self, pool):
        for key in pool.descriptorNames():
            x = pool[key]
            if type(x) is float:
                self.assertValidNumber(x)
            elif type(x) is numpy.ndarray:
                self.assert_(not numpy.isnan(x).any())
                self.assert_(not numpy.isinf(x).any())

    def assertEqualVector(self, found, expected):
        self.assertEqual(len(found), len(expected))
        for val1, val2 in zip(found, expected):
            self.assertEqual(val1, val2)

    def assertEqualMatrix(self, found, expected):
        self.assertEqual(len(found), len(expected))
        for v1, v2 in zip(found, expected):
            self.assertEqual(len(v1), len(v2))
            for val1, val2 in zip(v1, v2):
                if (isinstance(val1, numpy.ndarray)):
                    self.assertEqual(val1.all(), val2.all())
                else:
                    self.assertEqual(val1, val2)

    def assertAlmostEqualFixedPrecision(self, found, expected, digits=0):
        BaseTestCase.assertAlmostEqual(self, found, expected, digits)

    def assertAlmostEqualVectorFixedPrecision(self, found, expected, digits=0):
        self.assertEqual(len(found), len(expected))
        for val1, val2 in zip(found, expected):
            self.assertAlmostEqualFixedPrecision(val1, val2, digits)

    def assertAlmostEqual(self, found, expected, precision=1e-7):
        if expected == 0:
            diff = abs(found)
        elif found == 0:
            diff = abs(expected)
        else:
            diff = abs((expected - found) / abs(expected))
        self.assertTrue(diff <= precision,
                     """Difference is %e while allowed relative error is %e""" % (diff, precision))

    def assertAlmostEqualVector(self, found, expected, precision=1e-7):
        # we can use the optimized version if the two arrays are 1D numpy float arrays
        if isinstance(found, numpy.ndarray) and \
                isinstance(expected, numpy.ndarray) and \
                found.ndim == 1 and expected.ndim == 1 and \
                found.dtype == expected.dtype and \
                found.dtype == numpy.float32:
            return self.assertTrue(almostEqualArray(found, expected, precision))

        self.assertEqual(len(found), len(expected))
        for val1, val2 in zip(found, expected):
            self.assertAlmostEqual(val1, val2, precision)

    def assertAlmostEqualMatrix(self, found, expected, precision=1e-7):
        # we can use the optimized version if the two arrays are 2D numpy float arrays
        if isinstance(found, numpy.ndarray) and \
                isinstance(expected, numpy.ndarray) and \
                found.ndim == 2 and expected.ndim == 2 and \
                found.dtype == expected.dtype and \
                found.dtype == numpy.float64:
            return self.assertTrue(almostEqualArray(found, expected, precision))

        self.assertEqual(len(found), len(expected))

        for v1, v2 in zip(found, expected):
            self.assertEqual(len(v1), len(v2))
            self.assertAlmostEqualVector(np.array(v1).flatten(), np.array(v2).flatten(), precision)

    def assertAlmostEqualAbs(self, found, expected, precision=0.1):
        diff = abs(expected - found)
        self.assert_(diff <= precision, 'Difference is %e while allowed absolute error is %e' % (diff, precision))

    def assertAlmostEqualVectorAbs(self, found, expected, precision=0.1):
        self.assertEqual(len(found), len(expected))
        for val1, val2 in zip(found, expected):
            self.assertAlmostEqualAbs(val1, val2, precision)

    def assertConfigureFails(self, algo, params):
        conf = lambda: algo.configure(**params)
        self.assertRaises(EssentiaException, conf)

    def assertConfigureSuccess(self, algo, params):
        try:
            algo.configure(**params)
        except EssentiaException:
            self.fail()

    def assertComputeSuccess(self, algo, params):
        try:
            algo.compute(**params)
        except EssentiaException:
            self.fail()

    def assertComputeFails(self, algo, *args):
        comp = lambda: algo.compute(*args)
        self.assertRaises(EssentiaException, comp)


def almostEqualArray(found, expected, precision):
    assert np.shape(found) == np.shape(expected)

    for i in range(np.shape(found)[0]):
        for j in range(np.shape(found)[1]):
            x = found[i][j]
            y = expected[i][j]
            if y == 0:
                diff = abs(x)
            elif x == 0:
                diff = abs(y)
            else:
                diff = abs((y - x) / abs(y))
            if diff > precision:
                return False
    return True


def almostEqualAudioArray(found, expected, precision):
    return False


TestLoader.__test__ = False
TestSuite.__test__ = False
TestResult.__test__ = False
