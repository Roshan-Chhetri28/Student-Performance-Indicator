from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requrements(file_path:str)->List[str]:
    '''This function returns list of requirments'''
    requirements = []
    with open('requirements.txt') as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements



setup(
    name= 'ML project',
    version='0.0.1',
    author='Roshan',
    author_email='roshanchhetri931@gmail.com',
    packages=find_packages(),
    install_requires = get_requrements('requirements.txt')
)
