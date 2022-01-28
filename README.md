# Motion-Detector
This app detects moving object in front of computer webcam.
It also writes the rectangle on a moving object.
It records the times when the movement starts and the times when the movement ends.
It also saves the video with the movement to 'output.avi'.

Libraries cv2 and pandas were used.


In case of using pylint we get may errors like:
E1101: Module 'cv2' has no 'imshow' member (no-member)

To avoid that generate a pylint config file in the root of your project with this command: 
pylint --generate-rcfile > .pylintrc 
(To generate in the home directory: pylint --generate-rcfile > ~/.pylintrc)

At the beginning of the generated .pylintrc file you will see:
# A comma-separated list of package or module names from where C extensions may
# be loaded. Extensions are loading into the active Python interpreter and may
# run arbitrary code.
extension-pkg-whitelist=

Add cv2 so you end up with:
# A comma-separated list of package or module names from where C extensions may
# be loaded. Extensions are loading into the active Python interpreter and may
# run arbitrary code.
extension-pkg-whitelist=cv2

Save the file. The lint errors should disappear.

