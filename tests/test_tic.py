#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for `tic` package.'''

import pytest


import tic


@pytest.fixture
def author():
    '''Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    '''
    return tic.__author__


def test_content(author):
    '''Sample pytest test function with the pytest fixture as an argument.'''
    assert 'Tom Martensen' == author
