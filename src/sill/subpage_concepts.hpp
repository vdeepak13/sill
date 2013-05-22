/* subpage_concepts.hpp
 * This file holds documentation for the concepts we use.
 */

/**
 * \page subpage_concepts Concept Reference
 *
 * \ref index "Back to Main Page"
 *
 * \section concepts_prl PRL Concepts
 * See the \ref group_concepts "PRL Concepts module" for a list of concepts
 *   and links to documentation.
 *
 * \section concepts_outside Concepts from Outside Libraries
 *
 * \todo Remove places in the code where **Concept concept labels are used
 *   (as opposed to **); e.g. SinglePassRangeConcept should be changed to
 *   SinglePassRange.  The **Concept labeling is deprecated.
 *
 * \todo Make the template parameters for these concepts generic. (The
 *   ones used in the code are there now.)
 *
 * \subsection concepts_boost_graph Graph Concepts
 * See: prl/detail/graph_concepts.hpp
 * - Imports all of 'boost::::concepts' in boost/graph/graph_concepts.hpp
 *    into prl.
 *
 * Concepts used in PRL:
 * - Graph<G>
 * - EdgeMutableGraph<G>
 * - AdjacencyGraph<G>
 * - IncidenceGraph<G>
 * - VertexListGraph<Graph>
 * - VertexAndEdgeListGraph<G>
 * - MutableGraph<G>
 *
 * \subsection concepts_pstade Range Concepts
 * See: prl/detail/range_concepts.hpp
 * - Imports a bunch of concepts from pstade/oven/concepts.hpp into prl.
 * - Renames 'pstade::::oven' as 'range'
 *
 * Concepts used in PRL:
 * - ReadableForwardRange<Range>
 * - ReadableForwardRangeConvertible<Range, variable_h>
 * - SinglePassRange<Range>
 * - InputRange<R>
 * - InputRangeConvertible<IndexList,index>
 * - InputRangeConvertible<Range,variable_h>
 * - InputRangeOver<R, variable_h>
 * - InputRandomAccessRangeC<Range, variable_h>
 * - ForwardRangeC<CliqueRange, domain>
 * - InputIterator<It>
 * - OutputIterator<OutIt, size_type>
 * - ReadableIterator<It>
 * - ForwardTraversal<It>
 * - ForwardRangeNoDefault<Range>
 *
 * \subsection concepts_boost_stl STL Concepts
 * See: prl/stl_concepts.hpp
 * - Imports a bunch of concepts from boost/concept_check.hpp into prl
 *
 * Concepts used in PRL: (NOTE: Some of these might belong elsewhere.)
 * - BinaryFunction<Op,T,T,T>
 * - DefaultConstructible<T>
 * - PairUniqueAssociativeContainer<Map>
 * - Assignable<T>
 * - EqualityComparable<T>
 * - LessThanComparable<T>
 * - UnaryPredicate<Predicate, Set>
 * - Convertible<OtherT,T>
 * - Invertible<factor_t>
 * - ReadablePropertyMapConcept<PMap, vertex_type>
 *
 * \todo Is DistributionFactor<factor_t> deprecated?
 */
