.. _ref_quickstart:

Quickstart
==========

.. caution::
   This documentation is still under development.

To install the **OxyGenie** package and start simulating vascular networks, follow the steps below:

.. _installation:

Installation
------------

.. dropdown:: 1. **Prerequisites**   :octicon:`code`
    :animate: fade-in-slide-down

    Ensure that Python >3.9 is installed on your system.  
    If you haven't installed Python, you can download it from the `official Python website <https://www.python.org>`_.

.. dropdown:: 2. **Create a virtual environment (Optional)** :octicon:`code`
    :animate: fade-in-slide-down

    It is recommended to create a virtual environment to isolate project dependencies.  
    Open a terminal or command prompt and run the following commands:

    .. tab-set::

        .. tab-item:: On macOS and Linux:

            .. code-block:: console
        
                python3 -m venv myenv
                source myenv/bin/activate

        .. tab-item:: On Windows:

            .. code-block:: console

                python -m venv myenv
                myenv\Scripts\activate

.. dropdown:: 3. **Install the required packages** :octicon:`code`
    :open:
    :animate: fade-in-slide-down
       
    In the virtual environment or your Python environment,  
    run the following command to install the necessary packages:

    .. code-block:: console

        pip install -r requirements.txt

.. dropdown:: 4. **Install the OxyGenie package** :octicon:`code`
    :open:
    :animate: fade-in-slide-down

    Download the OxyGenie repository from the source code available on GitHub.

.. dropdown:: 5. **Import the library** :octicon:`code`
    :open:
    :animate: fade-in-slide-down

    Once the library is downloaded, navigate to the project directory. At the root of the project, you can use the following code to import the package:

    .. code-block:: python

        from OxyGenie.diffusion import *
        import OxyGenie.pgvnet as pvg
        from OxyGenie.learn import *

.. tip::

    Now that you have installed all the required prerequisites and dependencies,  
    feel free to consult the library documentation for more information on  
    usage and the various functions available.
