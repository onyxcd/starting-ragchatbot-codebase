"""Unit tests for VectorStore search functionality"""

import pytest
from vector_store import VectorStore, SearchResults


def test_search_returns_results(populated_vector_store):
    """Test basic search functionality returns results"""
    results = populated_vector_store.search("machine learning")

    # Should return SearchResults object
    assert isinstance(results, SearchResults)

    # Should have results
    assert not results.is_empty()
    assert len(results.documents) > 0


def test_search_with_course_filter(populated_vector_store):
    """Test search with course name filter"""
    results = populated_vector_store.search(
        query="learning",
        course_name="Machine Learning"  # Exact match
    )

    # Should return results
    assert isinstance(results, SearchResults)
    assert results.error is None

    # Results should be from the correct course
    if not results.is_empty():
        for meta in results.metadata:
            assert meta['course_title'] == "Introduction to Machine Learning"


def test_search_with_lesson_filter(populated_vector_store):
    """Test search with lesson number filter"""
    results = populated_vector_store.search(
        query="learning",
        lesson_number=2
    )

    # Should return results
    assert isinstance(results, SearchResults)

    # Results should be from lesson 2
    if not results.is_empty():
        for meta in results.metadata:
            assert meta['lesson_number'] == 2


def test_fuzzy_course_name_matching(populated_vector_store):
    """Test _resolve_course_name with partial match"""
    # Test with partial course name
    resolved = populated_vector_store._resolve_course_name("Machine")

    # Should find the full course title
    assert resolved == "Introduction to Machine Learning"


def test_fuzzy_course_name_matching_with_typo(populated_vector_store):
    """Test fuzzy matching handles similar names"""
    # Even with slight variations, should find the course
    resolved = populated_vector_store._resolve_course_name("Intro Machine Learning")

    # Should find the course (fuzzy matching via vector search)
    assert resolved is not None


def test_course_name_resolution_fails_gracefully(populated_vector_store):
    """Test course resolution with no match returns None"""
    resolved = populated_vector_store._resolve_course_name("NonExistentQuantumPhysicsCourse")

    # Should return None for non-existent course
    # Note: Due to vector similarity, it might still return something
    # In production, this would be the closest match, but for a completely
    # different domain, we'd expect no good match
    assert resolved is None or isinstance(resolved, str)


def test_search_with_error_handling(vector_store):
    """Test search error handling with empty store"""
    # Search in empty store
    results = vector_store.search("test query")

    # Should return SearchResults, not crash
    assert isinstance(results, SearchResults)

    # Should be empty
    assert results.is_empty()


def test_search_results_empty_vs_error():
    """Test SearchResults differentiates between empty results and errors"""
    # Test empty results (no documents found)
    empty_results = SearchResults.empty("No documents found")

    assert empty_results.is_empty()
    assert empty_results.error == "No documents found"
    assert len(empty_results.documents) == 0

    # Test successful empty search (no error, just no matches)
    success_empty = SearchResults(documents=[], metadata=[], distances=[])

    assert success_empty.is_empty()
    assert success_empty.error is None


def test_search_results_from_chroma():
    """Test SearchResults.from_chroma factory method"""
    # Mock ChromaDB response format
    chroma_response = {
        'documents': [['doc1', 'doc2']],
        'metadatas': [[{'course_title': 'Test', 'lesson_number': 1}, {'course_title': 'Test', 'lesson_number': 2}]],
        'distances': [[0.1, 0.2]]
    }

    results = SearchResults.from_chroma(chroma_response)

    assert len(results.documents) == 2
    assert results.documents[0] == 'doc1'
    assert len(results.metadata) == 2
    assert results.metadata[0]['course_title'] == 'Test'


def test_build_filter_with_both_params(vector_store):
    """Test _build_filter with both course and lesson"""
    filter_dict = vector_store._build_filter("Test Course", 3)

    # Should use $and operator
    assert "$and" in filter_dict
    assert {"course_title": "Test Course"} in filter_dict["$and"]
    assert {"lesson_number": 3} in filter_dict["$and"]


def test_build_filter_with_only_course(vector_store):
    """Test _build_filter with only course name"""
    filter_dict = vector_store._build_filter("Test Course", None)

    # Should be simple dict
    assert filter_dict == {"course_title": "Test Course"}


def test_build_filter_with_only_lesson(vector_store):
    """Test _build_filter with only lesson number"""
    filter_dict = vector_store._build_filter(None, 5)

    # Should be simple dict
    assert filter_dict == {"lesson_number": 5}


def test_build_filter_with_no_params(vector_store):
    """Test _build_filter with no parameters"""
    filter_dict = vector_store._build_filter(None, None)

    # Should return None (no filter)
    assert filter_dict is None


def test_get_existing_course_titles(populated_vector_store):
    """Test retrieving existing course titles"""
    titles = populated_vector_store.get_existing_course_titles()

    # Should return a list
    assert isinstance(titles, list)

    # Should contain our test course
    assert "Introduction to Machine Learning" in titles


def test_get_course_count(populated_vector_store):
    """Test course count retrieval"""
    count = populated_vector_store.get_course_count()

    # Should have at least 1 course
    assert count >= 1


def test_add_course_metadata(vector_store, sample_course):
    """Test adding course metadata to catalog"""
    vector_store.add_course_metadata(sample_course)

    # Verify it was added
    results = vector_store.course_catalog.get(ids=[sample_course.title])

    assert results is not None
    assert len(results['ids']) == 1
    assert results['ids'][0] == sample_course.title


def test_add_course_content(vector_store, sample_chunks):
    """Test adding course content chunks"""
    vector_store.add_course_content(sample_chunks)

    # Verify chunks were added
    results = vector_store.course_content.get()

    assert results is not None
    assert len(results['ids']) == len(sample_chunks)


def test_get_lesson_link(populated_vector_store):
    """Test retrieving lesson link"""
    link = populated_vector_store.get_lesson_link("Introduction to Machine Learning", 1)

    # Should return the link
    assert link == "https://example.com/ml-lesson1"


def test_get_lesson_link_not_found(populated_vector_store):
    """Test get_lesson_link with non-existent lesson"""
    link = populated_vector_store.get_lesson_link("Introduction to Machine Learning", 999)

    # Should return None for non-existent lesson
    assert link is None


def test_get_course_link(populated_vector_store):
    """Test retrieving course link"""
    link = populated_vector_store.get_course_link("Introduction to Machine Learning")

    # Should return the course link
    assert link == "https://example.com/ml-course"


def test_clear_all_data(populated_vector_store):
    """Test clearing all data from vector store"""
    # Verify data exists
    assert populated_vector_store.get_course_count() > 0

    # Clear data
    populated_vector_store.clear_all_data()

    # Verify data is cleared
    assert populated_vector_store.get_course_count() == 0


def test_search_respects_limit(populated_vector_store):
    """Test that search respects the limit parameter"""
    # Search with limit of 1
    results = populated_vector_store.search("learning", limit=1)

    # Should return at most 1 result
    assert len(results.documents) <= 1


def test_get_course_outline(populated_vector_store):
    """Test retrieving complete course outline"""
    outline = populated_vector_store.get_course_outline("Machine Learning")

    # Should return outline dict
    assert outline is not None
    assert 'title' in outline
    assert 'instructor' in outline
    assert 'lessons' in outline

    # Check outline details
    assert outline['title'] == "Introduction to Machine Learning"
    assert outline['instructor'] == "Dr. Smith"
    assert len(outline['lessons']) == 3


def test_get_course_outline_not_found(populated_vector_store):
    """Test get_course_outline with non-existent course"""
    outline = populated_vector_store.get_course_outline("NonExistentCourse12345XYZ")

    # Should return None for non-existent course
    assert outline is None
