import sys
import os

# Tambahkan path ke direktori yang berisi dl_model.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'utils', 'model'))

from dl_model import BugPriorityClassifier

classifier = BugPriorityClassifier()
result = classifier.debug_prediction("Unable to add text . text boxes to any header/footer in Base")

print("\nHasil return:")
print(result)
