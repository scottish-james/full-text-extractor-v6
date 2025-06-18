# Add this to a test script or run in your environment

from src.processors.powerpoint import PowerPointProcessor

# First, add the debug method to PowerPointProcessor class
# (Copy the debug_duplication_source method from my previous artifact into powerpoint_processor.py)

# Then run this:
processor = PowerPointProcessor()

# Debug the first slide
print("=== DEBUGGING SLIDE 1 ===")
processor.debug_duplication_source("/Users/jamestaylor/Downloads/testing_powerpoint_v9.5.pptx", 1)

# If you want to check multiple slides:
print("\n" + "="*50)
print("=== DEBUGGING SLIDE 2 ===")
processor.debug_duplication_source("/Users/jamestaylor/Downloads/testing_powerpoint_v9.5.pptx", 2)

print("\n" + "="*50)
print("=== DEBUGGING SLIDE 3 ===")
processor.debug_duplication_source("/Users/jamestaylor/Downloads/testing_powerpoint_v9.5.pptx", 3)