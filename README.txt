ONTENTS OF THIS FILE
---------------------
 * Description
 * Installation
 * Execution
DESCRIPTION
------------
The pawpularity scoring streamlit site project allows users to upload a photo of their pet or a potential adopted pet and retrieve the predicted popularity of that photo.
The predicted score is attained by using the EfficientNet-B3 model with Pytorch to output a newly trained model with which to score the photos.
 * To try scoring a pet's photo visit:
   https://share.streamlit.io/mczerwin/cse6242_team135/app.py
INSTALLATION
------------
There is no installation needed to use the above url and the service is free.
All of the required packages can be found in the requirements.txt file and these were installed when our app was deployed in the streamlit service.
EXECUTION
-------------
Simply navigate to https://share.streamlit.io/mczerwin/cse6242_team135/app.py and either drag your photo onto the webpage or use the browse files option and navigate and select
the file of your photo to score.
To run locally:
1 - Activate a python environment.  You must use Python 3.7  or greater.
2 - In a command terminal type : run streamlit (path to project)/app.py
