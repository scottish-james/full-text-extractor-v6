# Add the debug method to your PowerPointProcessor class first
# Then run this code:

from src.processors.powerpoint import PowerPointProcessor

# Test the specific file
file_path = "/Users/jamestaylor/Downloads/testing_powerpoint_v9.7.pptx"

# Create processor instance
processor = PowerPointProcessor()

# First, let's see which slides are being detected as potential diagrams
print("🎯 RUNNING FULL PRESENTATION ANALYSIS")
print("=" * 60)

try:
    # Run the full conversion to see current diagram analysis
    result = processor.convert_pptx_to_markdown_enhanced(file_path)

    # Find the diagram analysis section
    if "DIAGRAM ANALYSIS" in result:
        lines = result.split('\n')
        in_diagram_section = False
        for line in lines:
            if "DIAGRAM ANALYSIS" in line:
                in_diagram_section = True
            if in_diagram_section:
                print(line)
                if line.strip() == "" and in_diagram_section and "Slide" in lines[lines.index(line) - 1]:
                    break

    print("\n" + "=" * 60)
    print("🔍 DETAILED DEBUGGING OF SLIDE 15 (Should be 95%)")
    print("=" * 60)

    # Now debug slide 15 specifically
    debug_results = processor.debug_shape_extraction(file_path, slide_number=15)

    if debug_results:
        print(f"\n📊 SUMMARY:")
        print(f"- Shapes on slide: {debug_results['total_shapes_on_slide']}")
        print(f"- After expansion: {debug_results['shapes_after_expansion']}")
        print(f"- Content blocks: {debug_results['content_blocks_created']}")
        print(f"- Lines detected: {debug_results['lines_detected']}")
        print(f"- Arrows detected: {debug_results['arrows_detected']}")
        print(f"- Final score: {debug_results['final_score']}")
        print(f"- Final probability: {debug_results['final_probability']}%")

        # Analyze the issue
        if debug_results['lines_detected'] == 0 and debug_results['arrows_detected'] == 0:
            print(f"\n❌ ISSUE IDENTIFIED:")
            print(f"No lines or arrows detected! This explains the 40% cap.")
            print(f"Expected: Lines and arrows should give 20+ points each")
            print(f"Missing points: 40+ points from line/arrow detection")

    print("\n" + "=" * 60)
    print("🔍 ALSO TESTING SLIDE 16 (Should be 95%)")
    print("=" * 60)

    # Also test slide 16
    debug_results_16 = processor.debug_shape_extraction(file_path, slide_number=16)

    if debug_results_16:
        print(f"\n📊 SLIDE 16 SUMMARY:")
        print(f"- Lines detected: {debug_results_16['lines_detected']}")
        print(f"- Arrows detected: {debug_results_16['arrows_detected']}")
        print(f"- Final probability: {debug_results_16['final_probability']}%")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

print(f"\n🎯 NEXT STEPS:")
print(f"1. Check the debug output above to see what shape types are found")
print(f"2. Look for any shapes that should be lines/arrows but aren't detected")
print(f"3. Check if group expansion is working properly")
print(f"4. See if AUTO_SHAPE arrow detection is working")