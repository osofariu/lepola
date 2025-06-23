import sqlite3
import json
from typing import Dict, List, Any, Optional

from src.core.config import Settings

settings = Settings()
con = sqlite3.connect("data/app.db", detect_types=sqlite3.PARSE_COLNAMES)


def get_analysis_results(filename: str) -> List[Dict[str, Any]]:
    """Get analysis results for a specific filename.

    Args:
        filename: The filename to search for

    Returns:
        List of analysis results
    """
    results = []
    cur = con.cursor()
    for row in cur.execute(
        """
        SELECT ar.id, ar.document_id, ar.confidence_level, ar.processing_time_ms, ar.model_used, 
               ar.warnings, ar.requires_human_review, ar.created_at, ar.updated_at 
        FROM analysis_results ar 
        JOIN documents d ON d.id = ar.document_id 
        WHERE d.filename = ?
        """,
        (filename,),
    ):
        (
            id,
            document_id,
            confidence_level,
            processing_time_ms,
            model_used,
            warnings,
            requires_human_review,
            created_at,
            updated_at,
        ) = row
        results.append(
            {
                "id": id,
                "document_id": document_id,
                "confidence_level": confidence_level,
                "processing_time_ms": processing_time_ms,
                "model_used": model_used,
                "warnings": warnings,
                "requires_human_review": requires_human_review,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
    return results


def get_all_document_data(filename: str) -> Dict[str, Any]:
    """Get all data related to a specific document filename across all tables.

    Args:
        filename: The filename to search for

    Returns:
        Dictionary containing all related data organized by table
    """
    cur = con.cursor()

    # First get the document
    cur.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
    document_row = cur.fetchone()

    if not document_row:
        return {"error": f"Document with filename '{filename}' not found"}

    # Get column names for documents table
    document_columns = [desc[0] for desc in cur.description]
    document_data = dict(zip(document_columns, document_row))
    document_id = document_data["id"]

    result = {
        "document": document_data,
        "document_metadata": [],
        "processing_logs": [],
        "analysis_results": [],
        "extracted_entities": [],
        "document_summaries": [],
        "key_provisions": [],
        "risk_assessments": [],
    }

    # Get document metadata
    cur.execute("SELECT * FROM document_metadata WHERE document_id = ?", (document_id,))
    metadata_columns = [desc[0] for desc in cur.description]
    for row in cur.fetchall():
        result["document_metadata"].append(dict(zip(metadata_columns, row)))

    # Get processing logs
    cur.execute("SELECT * FROM processing_logs WHERE document_id = ?", (document_id,))
    logs_columns = [desc[0] for desc in cur.description]
    for row in cur.fetchall():
        result["processing_logs"].append(dict(zip(logs_columns, row)))

    # Get analysis results
    cur.execute("SELECT * FROM analysis_results WHERE document_id = ?", (document_id,))
    analysis_columns = [desc[0] for desc in cur.description]
    analysis_ids = []
    for row in cur.fetchall():
        analysis_data = dict(zip(analysis_columns, row))
        result["analysis_results"].append(analysis_data)
        analysis_ids.append(analysis_data["id"])

    # For each analysis result, get related data
    for analysis_id in analysis_ids:
        # Get extracted entities
        cur.execute(
            "SELECT * FROM extracted_entities WHERE analysis_id = ?", (analysis_id,)
        )
        entities_columns = [desc[0] for desc in cur.description]
        for row in cur.fetchall():
            result["extracted_entities"].append(dict(zip(entities_columns, row)))

        # Get document summaries
        cur.execute(
            "SELECT * FROM document_summaries WHERE analysis_id = ?", (analysis_id,)
        )
        summaries_columns = [desc[0] for desc in cur.description]
        summary_ids = []
        for row in cur.fetchall():
            summary_data = dict(zip(summaries_columns, row))
            result["document_summaries"].append(summary_data)
            summary_ids.append(summary_data["id"])

        # For each summary, get provisions and risk assessments
        for summary_id in summary_ids:
            # Get key provisions
            cur.execute(
                "SELECT * FROM key_provisions WHERE summary_id = ?", (summary_id,)
            )
            provisions_columns = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                result["key_provisions"].append(dict(zip(provisions_columns, row)))

            # Get risk assessments
            cur.execute(
                "SELECT * FROM risk_assessments WHERE summary_id = ?", (summary_id,)
            )
            risks_columns = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                result["risk_assessments"].append(dict(zip(risks_columns, row)))

    return result


def deep_clear_file(filename: str, dry_run: bool = True) -> Dict[str, Any]:
    """Delete all data related to a specific document filename across all database tables.

    This function deletes in the correct order to respect foreign key constraints:
    1. risk_assessments (references document_summaries.id)
    2. key_provisions (references document_summaries.id)
    3. document_summaries (references analysis_results.id)
    4. extracted_entities (references analysis_results.id)
    5. analysis_results (references documents.id)
    6. processing_logs (references documents.id)
    7. document_metadata (references documents.id)
    8. documents (root table)

    Args:
        filename: The filename to delete all data for
        dry_run: If True, only shows what would be deleted without actually deleting

    Returns:
        Dictionary containing deletion results and counts
    """
    cur = con.cursor()

    # First check if document exists
    cur.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
    document_row = cur.fetchone()

    if not document_row:
        return {"error": f"Document with filename '{filename}' not found"}

    document_id = document_row[0]

    deletion_stats = {
        "filename": filename,
        "document_id": document_id,
        "dry_run": dry_run,
        "deleted_counts": {
            "risk_assessments": 0,
            "key_provisions": 0,
            "document_summaries": 0,
            "extracted_entities": 0,
            "analysis_results": 0,
            "processing_logs": 0,
            "document_metadata": 0,
            "documents": 0,
        },
        "operations": [],
    }

    try:
        # Get analysis result IDs for this document
        cur.execute(
            "SELECT id FROM analysis_results WHERE document_id = ?", (document_id,)
        )
        analysis_ids = [row[0] for row in cur.fetchall()]

        # Get summary IDs for these analysis results
        summary_ids = []
        for analysis_id in analysis_ids:
            cur.execute(
                "SELECT id FROM document_summaries WHERE analysis_id = ?",
                (analysis_id,),
            )
            summary_ids.extend([row[0] for row in cur.fetchall()])

        # 1. Delete risk_assessments
        for summary_id in summary_ids:
            cur.execute(
                "SELECT COUNT(*) FROM risk_assessments WHERE summary_id = ?",
                (summary_id,),
            )
            count = cur.fetchone()[0]
            if count > 0:
                if not dry_run:
                    cur.execute(
                        "DELETE FROM risk_assessments WHERE summary_id = ?",
                        (summary_id,),
                    )
                deletion_stats["deleted_counts"]["risk_assessments"] += count
                deletion_stats["operations"].append(
                    f"{'Would delete' if dry_run else 'Deleted'} {count} risk_assessments for summary_id {summary_id}"
                )

        # 2. Delete key_provisions
        for summary_id in summary_ids:
            cur.execute(
                "SELECT COUNT(*) FROM key_provisions WHERE summary_id = ?",
                (summary_id,),
            )
            count = cur.fetchone()[0]
            if count > 0:
                if not dry_run:
                    cur.execute(
                        "DELETE FROM key_provisions WHERE summary_id = ?", (summary_id,)
                    )
                deletion_stats["deleted_counts"]["key_provisions"] += count
                deletion_stats["operations"].append(
                    f"{'Would delete' if dry_run else 'Deleted'} {count} key_provisions for summary_id {summary_id}"
                )

        # 3. Delete document_summaries
        for analysis_id in analysis_ids:
            cur.execute(
                "SELECT COUNT(*) FROM document_summaries WHERE analysis_id = ?",
                (analysis_id,),
            )
            count = cur.fetchone()[0]
            if count > 0:
                if not dry_run:
                    cur.execute(
                        "DELETE FROM document_summaries WHERE analysis_id = ?",
                        (analysis_id,),
                    )
                deletion_stats["deleted_counts"]["document_summaries"] += count
                deletion_stats["operations"].append(
                    f"{'Would delete' if dry_run else 'Deleted'} {count} document_summaries for analysis_id {analysis_id}"
                )

        # 4. Delete extracted_entities
        for analysis_id in analysis_ids:
            cur.execute(
                "SELECT COUNT(*) FROM extracted_entities WHERE analysis_id = ?",
                (analysis_id,),
            )
            count = cur.fetchone()[0]
            if count > 0:
                if not dry_run:
                    cur.execute(
                        "DELETE FROM extracted_entities WHERE analysis_id = ?",
                        (analysis_id,),
                    )
                deletion_stats["deleted_counts"]["extracted_entities"] += count
                deletion_stats["operations"].append(
                    f"{'Would delete' if dry_run else 'Deleted'} {count} extracted_entities for analysis_id {analysis_id}"
                )

        # 5. Delete analysis_results
        cur.execute(
            "SELECT COUNT(*) FROM analysis_results WHERE document_id = ?",
            (document_id,),
        )
        count = cur.fetchone()[0]
        if count > 0:
            if not dry_run:
                cur.execute(
                    "DELETE FROM analysis_results WHERE document_id = ?", (document_id,)
                )
            deletion_stats["deleted_counts"]["analysis_results"] = count
            deletion_stats["operations"].append(
                f"{'Would delete' if dry_run else 'Deleted'} {count} analysis_results for document_id {document_id}"
            )

        # 6. Delete processing_logs
        cur.execute(
            "SELECT COUNT(*) FROM processing_logs WHERE document_id = ?", (document_id,)
        )
        count = cur.fetchone()[0]
        if count > 0:
            if not dry_run:
                cur.execute(
                    "DELETE FROM processing_logs WHERE document_id = ?", (document_id,)
                )
            deletion_stats["deleted_counts"]["processing_logs"] = count
            deletion_stats["operations"].append(
                f"{'Would delete' if dry_run else 'Deleted'} {count} processing_logs for document_id {document_id}"
            )

        # 7. Delete document_metadata
        cur.execute(
            "SELECT COUNT(*) FROM document_metadata WHERE document_id = ?",
            (document_id,),
        )
        count = cur.fetchone()[0]
        if count > 0:
            if not dry_run:
                cur.execute(
                    "DELETE FROM document_metadata WHERE document_id = ?",
                    (document_id,),
                )
            deletion_stats["deleted_counts"]["document_metadata"] = count
            deletion_stats["operations"].append(
                f"{'Would delete' if dry_run else 'Deleted'} {count} document_metadata for document_id {document_id}"
            )

        # 8. Delete documents
        if not dry_run:
            cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        deletion_stats["deleted_counts"]["documents"] = 1
        deletion_stats["operations"].append(
            f"{'Would delete' if dry_run else 'Deleted'} 1 document with filename {filename}"
        )

        # Commit if not dry run
        if not dry_run:
            con.commit()
            deletion_stats["status"] = "completed"
        else:
            deletion_stats["status"] = "dry_run_completed"

    except Exception as e:
        if not dry_run:
            con.rollback()
        deletion_stats["status"] = "error"
        deletion_stats["error"] = str(e)
        deletion_stats["operations"].append(f"Error: {str(e)}")

    return deletion_stats


def get_document_summary(filename: str) -> Dict[str, Any]:
    """Get a summary of data counts for a specific document filename.

    Args:
        filename: The filename to get summary for

    Returns:
        Dictionary containing counts of related data
    """
    cur = con.cursor()

    # Check if document exists
    cur.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
    document_row = cur.fetchone()

    if not document_row:
        return {"error": f"Document with filename '{filename}' not found"}

    document_id = document_row[0]

    summary = {"filename": filename, "document_id": document_id, "counts": {}}

    # Count records in each table
    tables_and_conditions = [
        ("documents", "id = ?", [document_id]),
        ("document_metadata", "document_id = ?", [document_id]),
        ("processing_logs", "document_id = ?", [document_id]),
        ("analysis_results", "document_id = ?", [document_id]),
    ]

    # Get analysis IDs for further queries
    cur.execute("SELECT id FROM analysis_results WHERE document_id = ?", (document_id,))
    analysis_ids = [row[0] for row in cur.fetchall()]

    for analysis_id in analysis_ids:
        tables_and_conditions.extend(
            [
                ("extracted_entities", "analysis_id = ?", [analysis_id]),
                ("document_summaries", "analysis_id = ?", [analysis_id]),
            ]
        )

    # Get summary IDs for further queries
    summary_ids = []
    for analysis_id in analysis_ids:
        cur.execute(
            "SELECT id FROM document_summaries WHERE analysis_id = ?", (analysis_id,)
        )
        summary_ids.extend([row[0] for row in cur.fetchall()])

    for summary_id in summary_ids:
        tables_and_conditions.extend(
            [
                ("key_provisions", "summary_id = ?", [summary_id]),
                ("risk_assessments", "summary_id = ?", [summary_id]),
            ]
        )

    # Count records in each table
    for table, condition, params in tables_and_conditions:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {condition}", params)
        count = cur.fetchone()[0]
        if table in summary["counts"]:
            summary["counts"][table] += count
        else:
            summary["counts"][table] = count

    return summary


if __name__ == "__main__":
    import sys

    # Accept input from the user
    input_args = sys.argv[1:]

    if not input_args:
        # No arguments provided, list existing document names with their creation dates
        cur = con.cursor()
        cur.execute("SELECT filename, id, created_at FROM documents")
        documents = cur.fetchall()

        if documents:
            print("=== Available Documents ===")
            for filename, id, created_at in documents:
                print(f"Filename: {filename}, ID: {id}, Created At: {created_at}")
        else:
            print("No documents found in the database.")
        sys.exit(0)

    # Check if the last argument is "execute" to determine dry-run status
    dry_run = True
    if input_args and input_args[-1].lower() == "execute":
        dry_run = False
        input_args = input_args[:-1]  # Remove "execute" from the list

    # Process each filename
    for filename in input_args:
        print(f"=== Analysis Results for {filename} ===")
        analysis_results = get_analysis_results(filename)
        print(json.dumps(analysis_results, indent=2))

        print(f"\n=== Document Summary for {filename} ===")
        doc_summary = get_document_summary(filename)
        print(json.dumps(doc_summary, indent=2))

        print(
            f"\n=== Deep Clear {'(ACTUAL)' if not dry_run else 'Dry Run'} for {filename} ==="
        )
        result = deep_clear_file(filename, dry_run=dry_run)
        print(json.dumps(result, indent=2))
