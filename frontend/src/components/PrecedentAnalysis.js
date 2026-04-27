import React, { useState } from 'react';

const PrecedentAnalysis = ({ response, queryType }) => {
    const [expandedPrecedent, setExpandedPrecedent] = useState(null);

    if (queryType === 'general_query') {
        return (
            <div className="space-y-4">
                {/* <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-900 mb-2">Answer</h3>
                    <p className="text-gray-700 text-sm leading-relaxed">{response.answer}</p>
                </div> */}

                {response.supporting_documents && response.supporting_documents.length > 0 && (
                    <div className="bg-white border border-gray-200 rounded-lg p-4">
                        <h3 className="font-semibold text-gray-900 mb-3">Supporting Documents</h3>
                        <div className="space-y-2">
                            {response.supporting_documents.map((doc, idx) => (
                                <div
                                    key={idx}
                                    className="border-l-4 border-green-400 pl-3 py-2"
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <p className="font-medium text-gray-900 text-sm">
                                                {doc.case_name || doc.file_name}
                                            </p>
                                            <p className="text-gray-600 text-xs mt-1">
                                                Relevance: {(doc.relevance_score * 100).toFixed(1)}%
                                            </p>
                                            {doc.excerpt && (
                                                <p className="text-gray-600 text-xs mt-2 italic">
                                                    "{doc.excerpt.substring(0, 120)}..."
                                                </p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        );
    }

    // Research query mode
    return (
        <div className="space-y-4">
            {/* Supporting Precedents */}
            {response.supporting_precedents && response.supporting_precedents.length > 0 && (
                <div className="bg-white border border-green-200 rounded-lg p-4">
                    <h3 className="font-semibold text-green-900 mb-3 flex items-center gap-2">
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                clipRule="evenodd"
                            />
                        </svg>
                        Supporting Precedents ({response.supporting_precedents.length})
                    </h3>
                    <div className="space-y-2">
                        {response.supporting_precedents.map((prec, idx) => (
                            <div key={idx} className="border-l-4 border-green-400 pl-3 py-2">
                                <button
                                    onClick={() =>
                                        setExpandedPrecedent(expandedPrecedent === `support-${idx}` ? null : `support-${idx}`)
                                    }
                                    className="w-full text-left hover:bg-gray-50 p-2 rounded -m-2"
                                >
                                    <p className="font-medium text-gray-900 text-sm">
                                        {prec.case_name || prec.file_name}
                                    </p>
                                    <p className="text-gray-600 text-xs mt-1">{prec.legal_principle}</p>
                                </button>
                                {expandedPrecedent === `support-${idx}` && (
                                    <div className="mt-2 p-2 bg-green-50 rounded text-xs text-gray-700 space-y-2">
                                        <div>
                                            <p className="font-semibold text-green-900">Factual Alignment:</p>
                                            <p>{prec.factual_alignment}</p>
                                        </div>
                                        {prec.excerpt && (
                                            <div>
                                                <p className="font-semibold text-green-900">Excerpt:</p>
                                                <p className="italic">"{prec.excerpt}"</p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Adverse Precedents */}
            {response.adverse_precedents && response.adverse_precedents.length > 0 && (
                <div className="bg-white border border-red-200 rounded-lg p-4">
                    <h3 className="font-semibold text-red-900 mb-3 flex items-center gap-2">
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                                clipRule="evenodd"
                            />
                        </svg>
                        Adverse Precedents ({response.adverse_precedents.length})
                    </h3>
                    <div className="space-y-2">
                        {response.adverse_precedents.map((prec, idx) => (
                            <div key={idx} className="border-l-4 border-red-400 pl-3 py-2">
                                <button
                                    onClick={() =>
                                        setExpandedPrecedent(expandedPrecedent === `adverse-${idx}` ? null : `adverse-${idx}`)
                                    }
                                    className="w-full text-left hover:bg-gray-50 p-2 rounded -m-2"
                                >
                                    <p className="font-medium text-gray-900 text-sm">
                                        {prec.case_name || prec.file_name}
                                    </p>
                                    <p className="text-gray-600 text-xs mt-1">{prec.risk_description}</p>
                                </button>
                                {expandedPrecedent === `adverse-${idx}` && (
                                    <div className="mt-2 p-2 bg-red-50 rounded text-xs text-gray-700 space-y-2">
                                        <div>
                                            <p className="font-semibold text-red-900">Distinguishing Argument:</p>
                                            <p>{prec.distinguishing_argument}</p>
                                        </div>
                                        {prec.excerpt && (
                                            <div>
                                                <p className="font-semibold text-red-900">Excerpt:</p>
                                                <p className="italic">"{prec.excerpt}"</p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Strategy Recommendation */}
            {response.strategy_recommendation && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                    <h3 className="font-semibold text-indigo-900 mb-3">Strategy Recommendation</h3>

                    {response.strategy_recommendation.priority_arguments && (
                        <div className="mb-3">
                            <p className="text-sm font-medium text-indigo-900 mb-2">Priority Arguments:</p>
                            <ul className="space-y-1">
                                {response.strategy_recommendation.priority_arguments.map((arg, idx) => (
                                    <li key={idx} className="text-xs text-gray-700 flex gap-2">
                                        <span className="text-indigo-600">•</span>
                                        {arg}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {response.strategy_recommendation.compensation_range && (
                        <div className="mb-3">
                            <p className="text-sm font-medium text-indigo-900">Compensation Range:</p>
                            <p className="text-xs text-gray-700 mt-1">
                                {response.strategy_recommendation.compensation_range}
                            </p>
                        </div>
                    )}

                    {response.strategy_recommendation.risks && (
                        <div>
                            <p className="text-sm font-medium text-indigo-900 mb-2">Key Risks:</p>
                            <ul className="space-y-1">
                                {response.strategy_recommendation.risks.map((risk, idx) => (
                                    <li key={idx} className="text-xs text-gray-700 flex gap-2">
                                        <span className="text-red-600">⚠</span>
                                        {risk}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default PrecedentAnalysis;
