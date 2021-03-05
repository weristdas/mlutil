from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='mlutil',
   version='0.1',
   description='Util for ML',
   author='Neil Jie Yan',
   author_email='yanjie@ict.ac.cn, jiey@msr',
   packages=['mlutil'],
   url="http://weristdas",
   install_requires=['numpy', 'pandas', 'pykalman'], #external dependent packages
)
