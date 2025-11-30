from setuptools import setup  # , find_packages

ver_str = '0.0.1'

setup(
    name='my-py-lib',
    version=ver_str,
    description='Personally used python toolkit.',
    author='onesixth',
    author_email='noexist@noexist.noexist',
    url='https://github.com/One-sixth/my-py-lib',
    install_requires=[],
    entry_points={'console_scripts': []},
    packages=['my_py_lib', 'my_py_lib/dataset'],
    package_data={},
)
