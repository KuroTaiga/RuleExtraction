import json
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
from datetime import datetime

class ExerciseAnalysisReporter:
    """Reporter class for exercise analysis results."""
    
    def __init__(self):
        self.results = []
        self.summary_stats = {
            'total_videos': 0,
            'correct_top1': 0,
            'correct_top3': 0,
            'average_rank': 0.0
        }
        
    def add_result(self, video_name: str, extracted_rules: Dict, 
                  top_matches: List[Tuple[str, float]]):
        """
        Add a single video analysis result.
        
        Args:
            video_name: Name of the analyzed video (expected exercise)
            extracted_rules: Dictionary of extracted rules
            top_matches: List of (exercise_name, similarity_score) tuples
        """
        # Find rank of correct exercise
        correct_rank = None
        for i, (exercise, score) in enumerate(top_matches, 1):
            if exercise.lower() == video_name.lower():
                correct_rank = i
                break
                
        result = {
            'video_name': video_name,
            'extracted_rules': extracted_rules,
            'top_matches': top_matches,
            'correct_rank': correct_rank,
            'in_top1': correct_rank == 1 if correct_rank else False,
            'in_top3': correct_rank <= 3 if correct_rank else False,
            'correct_exercise_score': next((score for ex, score in top_matches 
                                         if ex.lower() == video_name.lower()), None)
        }
        
        self.results.append(result)
        
    def calculate_summary_stats(self):
        """Calculate summary statistics from all results."""
        total = len(self.results)
        if total == 0:
            return
            
        correct_top1 = sum(1 for r in self.results if r['in_top1'])
        correct_top3 = sum(1 for r in self.results if r['in_top3'])
        valid_ranks = [r['correct_rank'] for r in self.results if r['correct_rank'] is not None]
        
        self.summary_stats = {
            'total_videos': total,
            'correct_top1': correct_top1,
            'correct_top3': correct_top3,
            'top1_accuracy': correct_top1 / total if total > 0 else 0,
            'top3_accuracy': correct_top3 / total if total > 0 else 0,
            'average_rank': sum(valid_ranks) / len(valid_ranks) if valid_ranks else None,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
    def generate_detailed_report(self) -> str:
        """Generate a detailed text report of the analysis."""
        self.calculate_summary_stats()
        
        report = []
        report.append("=" * 80)
        report.append("EXERCISE ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        report.append("\nSUMMARY STATISTICS:")
        report.append(f"Total videos analyzed: {self.summary_stats['total_videos']}")
        report.append(f"Top-1 Accuracy: {self.summary_stats['top1_accuracy']:.2%}")
        report.append(f"Top-3 Accuracy: {self.summary_stats['top3_accuracy']:.2%}")
        if self.summary_stats['average_rank']:
            report.append(f"Average rank of correct exercise: {self.summary_stats['average_rank']:.2f}")
        
        # Detailed results
        report.append("\nDETAILED RESULTS:")
        for result in self.results:
            report.append("\n" + "-" * 40)
            report.append(f"Video: {result['video_name']}")
            
            if result['correct_rank']:
                report.append(f"Correct exercise rank: {result['correct_rank']}")
                report.append(f"Correct exercise similarity score: {result['correct_exercise_score']:.3f}")
            else:
                report.append("Correct exercise not found in matches")
            
            report.append("\nTop 3 Matches:")
            for i, (exercise, score) in enumerate(result['top_matches'][:3], 1):
                report.append(f"{i}. {exercise}: {score:.3f}")
        
        return "\n".join(report)
        
    def save_results(self, output_dir: str):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save the reports
        """
        timestamp = self.summary_stats['timestamp']
        
        # Save detailed report
        report_path = f"{output_dir}/exercise_analysis_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(self.generate_detailed_report())
            
        # Save raw results as JSON
        json_path = f"{output_dir}/exercise_analysis_data_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'summary_stats': self.summary_stats,
                'detailed_results': self.results
            }, f, indent=2)
            
        # Create Excel report
        excel_path = f"{output_dir}/exercise_analysis_{timestamp}.xlsx"
        self._save_excel_report(excel_path)
        
    def _save_excel_report(self, filepath: str):
        """Generate and save detailed Excel report."""
        # Create summary sheet data
        summary_data = {
            'Metric': ['Total Videos', 'Top-1 Accuracy', 'Top-3 Accuracy', 'Average Rank'],
            'Value': [
                self.summary_stats['total_videos'],
                self.summary_stats['top1_accuracy'],
                self.summary_stats['top3_accuracy'],
                self.summary_stats['average_rank']
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Create detailed results sheet data
        detailed_data = []
        for result in self.results:
            row = {
                'Video Name': result['video_name'],
                'Correct Rank': result['correct_rank'],
                'In Top-1': result['in_top1'],
                'In Top-3': result['in_top3'],
                'Correct Exercise Score': result['correct_exercise_score'],
            }
            # Add top 3 matches
            for i, (exercise, score) in enumerate(result['top_matches'][:3], 1):
                row[f'Match {i} Exercise'] = exercise
                row[f'Match {i} Score'] = score
            detailed_data.append(row)
        detailed_df = pd.DataFrame(detailed_data)
        
        # Save to Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)