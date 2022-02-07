//! Utilities for processing the ASTs provided by `tree_sitter`

use crate::diff::DiffEngine;
use crate::diff::Hunks;
use crate::diff::Myers;
use logging_timer::time;
use std::{cell::RefCell, ops::Index, path::PathBuf};
use tree_sitter::Node as TSNode;
use tree_sitter::Point;
use tree_sitter::Tree as TSTree;
use unicode_segmentation as us;

#[derive(Debug, Clone, Copy)]
pub struct EntryChar<'a> {
    /// The character that this entry represents
    pub c: &'a char,

    /// The position of the character
    pub position: Point,

    /// The position of the character as a byte offset from the beginning of the file.
    pub byte_offset: u32,
}

/// A mapping between a tree-sitter node and the text it corresponds to
#[derive(Debug, Clone, Copy)]
pub struct Entry<'a> {
    /// The node an entry in the diff vector refers to
    ///
    /// We keep a reference to the leaf node so that we can easily grab the text and other metadata
    /// surrounding the syntax
    pub reference: TSNode<'a>,

    /// A reference to the text the node refers to
    ///
    /// This is different from the `source_text` that the [AstVector](AstVector) refers to, as the
    /// entry only holds a reference to the specific range of text that the node covers.
    pub text: &'a str,

    /// An optional override for a node's start position
    pub start_position: Option<Point>,

    /// An optional override for a node's end position
    pub end_position: Option<Point>,
}

impl<'a> Entry<'a> {
    /// Split the text of an entry by line.
    pub fn split(self) -> Vec<Self> {
        let mut entries = Vec::new();

        // This will always reserve enough space, potentially more, because graphemes can be
        // multiple characters.
        entries.reserve(self.text.len());

        // TODO(afnan) you can figure out when we've moved onto a new line
        // if the column indicator resets. This isn't perfect, it won't work if
        // we go from a column on a previous line to a greater column on the next line
        let indices = us::UnicodeSegmentation::grapheme_indices(self.text, true);

        for (idx, grapheme) in indices {
            // Note that the indices here are offsets within the node text, so the column is the
            // starting column + current idx.
            let mut new_start_pos = self.reference.start_position();
            let original_start_col = self.reference.start_position().column;
            new_start_pos.column = original_start_col + idx;

            let mut new_end_pos = self.reference.start_position();
            // Every grapheme has to be a at least one byte
            debug_assert!(!grapheme.is_empty());

            // We substract one because these ranges are [inclusive, exclusive) to match the
            // tree-sitter indexing scheme.
            new_end_pos.column = original_start_col + idx + 1;

            debug_assert!(new_start_pos.column <= new_end_pos.column);
            debug_assert!(new_start_pos.row <= new_end_pos.row);

            let entry = Entry {
                reference: self.reference,
                text: &self.text[idx..=idx],
                start_position: Some(new_start_pos),
                end_position: Some(new_end_pos),
            };
            entries.push(entry);
        }

        entries
    }

    pub fn split_from_ast_vector(ast_vector: &'a AstVector<'a>) -> Vec<Self> {
        let mut entries = Vec::new();

        for entry in &ast_vector.leaves {
            entries.extend(entry.split().iter());
        }
        entries
    }

    /// Get the start position of an entry
    pub fn start_position(&self) -> Point {
        if let Some(pos) = self.start_position {
            pos
        } else {
            self.reference.start_position()
        }
    }

    /// Get the end position of an entry
    pub fn end_position(&self) -> Point {
        if let Some(pos) = self.end_position {
            pos
        } else {
            self.reference.end_position()
        }
    }
}

/// A vector that allows for linear traversal through the leafs of an AST.
///
/// This representation of the tree leaves is much more convenient for things like dynamic
/// programming, and provides useful for formatting.
#[derive(Debug)]
pub struct AstVector<'a> {
    /// The leaves of the AST, build with an in-order traversal
    pub leaves: Vec<Entry<'a>>,

    /// The full source text that the AST refers to
    pub source_text: &'a str,
}

impl<'a> Eq for Entry<'a> {}

/// A wrapper struct for AST vector data that owns the data that the AST vector references
///
/// Ideally we would just have the AST vector own the actual string and tree, but it makes things
/// extremely messy with the borrow checker, so we have this wrapper struct that holds the owned
/// data that the vector references. This gets tricky because the tree sitter library uses FFI so
/// the lifetime references get even more mangled.
#[derive(Debug)]
pub struct AstVectorData {
    /// The text in the file
    pub text: String,

    /// The tree that was parsed using the text
    pub tree: TSTree,

    /// The file path that the text corresponds to
    pub path: PathBuf,
}

impl<'a> AstVector<'a> {
    /// Create a `DiffVector` from a `tree_sitter` tree
    ///
    /// This method calls a helper function that does an in-order traversal of the tree and adds
    /// leaf nodes to a vector
    #[time("info", "ast::{}")]
    pub fn from_ts_tree(tree: &'a TSTree, text: &'a str) -> Self {
        let leaves = RefCell::new(Vec::new());
        build(&leaves, tree.root_node(), text);
        AstVector {
            leaves: leaves.into_inner(),
            source_text: text,
        }
    }

    /// Return the number of nodes in the diff vector
    pub fn len(&self) -> usize {
        self.leaves.len()
    }
}

impl<'a> Index<usize> for AstVector<'a> {
    type Output = Entry<'a>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.leaves[index]
    }
}

impl<'a> PartialEq for Entry<'a> {
    fn eq(&self, other: &Entry) -> bool {
        self.reference.kind_id() == other.reference.kind_id() && self.text == other.text
    }
}

impl<'a> PartialEq for AstVector<'a> {
    fn eq(&self, other: &AstVector) -> bool {
        if self.leaves.len() != other.leaves.len() {
            return false;
        }

        if self.leaves.len() != other.leaves.len() {
            return false;
        }

        for i in 0..self.leaves.len() {
            let leaf = self.leaves[i];
            let other_leaf = self.leaves[i];

            if leaf != other_leaf {
                return false;
            }
        }
        true
    }
}

/// Recursively build a vector from a given node
///
/// This is a helper function that simply walks the tree and collects leaves in an in-order manner.
/// Every time it encounters a leaf node, it stores the metadata and reference to the node in an
/// `Entry` struct.
fn build<'a>(vector: &RefCell<Vec<Entry<'a>>>, node: tree_sitter::Node<'a>, text: &'a str) {
    // If the node is a leaf, we can stop traversing
    if node.child_count() == 0 {
        // We only push an entry if the referenced text range isn't empty, since there's no point
        // in having an empty text range. This also fixes a bug where the program would panic
        // because it would attempt to access the 0th index in an empty text range.
        if !node.byte_range().is_empty() {
            let node_text: &'a str = &text[node.byte_range()];
            vector.borrow_mut().push(Entry {
                reference: node,
                text: node_text,
                start_position: None,
                end_position: None,
            });
        }
        return;
    }

    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        build(vector, child, text);
    }
}

/// The different types of elements that can be in an edit script
#[derive(Debug, Eq, PartialEq)]
pub enum EditType<T> {
    /// An element that was added in the edit script
    Addition(T),

    /// An element that was deleted in the edit script
    Deletion(T),
}

/// Compute the hunks corresponding to the minimum edit path between two documents
///
/// This method computes the minimum edit distance between two `DiffVector`s, which are the leaf
/// nodes of an AST, using the standard DP approach to the longest common subsequence problem, the
/// only twist is that here, instead of operating on raw text, we're operating on the leaves of an
/// AST.
///
/// This has O(mn) space complexity and uses O(mn) space to compute the minimum edit path, and then
/// has O(mn) space complexity and uses O(mn) space to backtrack and recreate the path.
///
/// This will return two groups of [hunks](diff::Hunks) in a tuple of the form
/// `(old_hunks, new_hunks)`.
#[time("info", "ast::{}")]
pub fn compute_edit_script<'a>(a: &'a AstVector, b: &'a AstVector) -> (Hunks<'a>, Hunks<'a>) {
    let myers = Myers::default();

    let a_characters = Entry::split_from_ast_vector(a);
    let b_characters = Entry::split_from_ast_vector(b);
    let edit_script = myers.diff(&a_characters[..], &b_characters[..]);
    let edit_script_len = edit_script.len();
    // TODO(afnan) convert ast vectors into ast

    // TODO(afnan): update the diff here, maybe update the leaves so we have per-character entries
    // so we can compute a diff on each character
    //let edit_script = myers.diff(&a.leaves[..], &b.leaves[..]);
    let mut old_edits = Vec::with_capacity(edit_script_len);
    let mut new_edits = Vec::with_capacity(edit_script_len);

    for edit in edit_script {
        match edit {
            EditType::Deletion(&e) => old_edits.push(e),
            EditType::Addition(&e) => new_edits.push(e),
        };
    }

    // Convert the vectors of edits into hunks that can be displayed
    let old_hunks = old_edits.into_iter().collect();
    let new_hunks = new_edits.into_iter().collect();
    (old_hunks, new_hunks)
}
