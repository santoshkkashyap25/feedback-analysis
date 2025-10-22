### benchmarks/performance_test.py

import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
from statistics import mean, median
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class APIPerformanceBenchmark:
    """Benchmark API performance under various loads"""
    
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.test_texts = [
            "This product is absolutely amazing! Best purchase ever.",
            "Terrible quality, completely disappointed with this item.",
            "Average product, nothing special but does the job.",
            "Outstanding service and great value for money!",
            "Poor packaging, item arrived damaged and unusable.",
            "Excellent quality and fast shipping, highly recommend.",
            "Not worth the money, found better alternatives elsewhere.",
            "Perfect for my needs, exactly as described.",
            "Customer service was unhelpful, product mediocre.",
            "Exceeded expectations, will definitely buy again!"
        ]
    
    def single_request_test(self, text):
        """Test single API request"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={'text': text},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'response_size': len(response.content)
                }
            else:
                return {
                    'success': False,
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'error': response.text
                }
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'response_time': end_time - start_time,
                'status_code': 0,
                'error': str(e)
            }
    
    def concurrent_load_test(self, num_requests=100, max_workers=10):
        """Test API under concurrent load"""
        print(f"Running concurrent load test: {num_requests} requests, {max_workers} workers")
        
        results = []
        texts = np.random.choice(self.test_texts, num_requests)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {
                executor.submit(self.single_request_test, text): text 
                for text in texts
            }
            
            for future in as_completed(future_to_text):
                result = future.result()
                results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            
            metrics = {
                'total_requests': num_requests,
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / num_requests * 100,
                'total_time': total_time,
                'requests_per_second': num_requests / total_time,
                'avg_response_time': mean(response_times),
                'median_response_time': median(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': np.percentile(response_times, 95),
                'p99_response_time': np.percentile(response_times, 99)
            }
        else:
            metrics = {
                'total_requests': num_requests,
                'successful_requests': 0,
                'failed_requests': num_requests,
                'success_rate': 0,
                'total_time': total_time,
                'requests_per_second': 0,
                'error': 'All requests failed'
            }
        
        return metrics, results
    
    def batch_request_test(self, batch_sizes=[1, 5, 10, 20, 50, 100]):
        """Test batch prediction endpoint with different batch sizes"""
        print("Running batch request test...")
        
        batch_results = []
        
        for batch_size in batch_sizes:
            texts = np.random.choice(self.test_texts, min(batch_size, len(self.test_texts)))
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}/predict/batch",
                    json={'texts': texts.tolist()},
                    headers={'Content-Type': 'application/json'},
                    timeout=60
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    batch_results.append({
                        'batch_size': batch_size,
                        'response_time': response_time,
                        'time_per_item': response_time / batch_size,
                        'success': True
                    })
                else:
                    batch_results.append({
                        'batch_size': batch_size,
                        'response_time': response_time,
                        'success': False,
                        'error': response.text
                    })
                    
            except Exception as e:
                batch_results.append({
                    'batch_size': batch_size,
                    'response_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                })
        
        return batch_results
    
    def stress_test(self, duration_seconds=60, concurrent_users=20):
        """Run stress test for specified duration"""
        print(f"Running stress test: {duration_seconds}s duration, {concurrent_users} concurrent users")
        
        end_time = time.time() + duration_seconds
        results = []
        
        def worker():
            while time.time() < end_time:
                text = np.random.choice(self.test_texts)
                result = self.single_request_test(text)
                results.append({**result, 'timestamp': time.time()})
                time.sleep(0.1)  # Small delay between requests
        
        # Start workers
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]
            
            # Wait for all workers to complete
            for future in as_completed(futures):
                future.result()
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            
            stress_metrics = {
                'duration': duration_seconds,
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(results) * 100,
                'requests_per_second': len(results) / duration_seconds,
                'avg_response_time': mean(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': np.percentile(response_times, 95),
                'error_rate': len(failed_requests) / len(results) * 100
            }
        else:
            stress_metrics = {
                'duration': duration_seconds,
                'total_requests': len(results),
                'successful_requests': 0,
                'failed_requests': len(results),
                'success_rate': 0,
                'requests_per_second': len(results) / duration_seconds,
                'error_rate': 100
            }
        
        return stress_metrics, results
    
    def generate_report(self, save_path='benchmark_report.txt'):
        """Generate comprehensive performance report"""
        print("Generating performance benchmark report")
        
        report_lines = []
        report_lines.append("CUSTOMER FEEDBACK API PERFORMANCE BENCHMARK REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"API Base URL: {self.base_url}")
        report_lines.append("")
        
        # Test 1: Single request baseline
        baseline_result = self.single_request_test(self.test_texts[0])
        report_lines.append("1. BASELINE SINGLE REQUEST TEST")
        report_lines.append("-" * 40)
        if baseline_result['success']:
            report_lines.append(f"Response Time: {baseline_result['response_time']:.3f}s")
            report_lines.append(f"Status Code: {baseline_result['status_code']}")
        else:
            report_lines.append(f"Failed: {baseline_result['error']}")
        report_lines.append("")
        
        # Test 2: Concurrent load test
        load_metrics, _ = self.concurrent_load_test(num_requests=100, max_workers=10)
        report_lines.append("2. CONCURRENT LOAD TEST (100 requests, 10 workers)")
        report_lines.append("-" * 40)
        report_lines.append(f"Success Rate: {load_metrics['success_rate']:.1f}%")
        report_lines.append(f"Requests/Second: {load_metrics['requests_per_second']:.2f}")
        if 'avg_response_time' in load_metrics:
            report_lines.append(f"Avg Response Time: {load_metrics['avg_response_time']:.3f}s")
            report_lines.append(f"P95 Response Time: {load_metrics['p95_response_time']:.3f}s")
            report_lines.append(f"P99 Response Time: {load_metrics['p99_response_time']:.3f}s")
        report_lines.append("")
        
        # Test 3: Batch request test
        batch_results = self.batch_request_test()
        report_lines.append("3. BATCH REQUEST TEST")
        report_lines.append("-" * 40)
        for result in batch_results:
            if result['success']:
                report_lines.append(f"Batch Size {result['batch_size']:3d}: "
                                  f"{result['response_time']:.3f}s "
                                  f"({result['time_per_item']:.3f}s per item)")
        report_lines.append("")
        
        # Test 4: Stress test
        stress_metrics, _ = self.stress_test(duration_seconds=30, concurrent_users=5)
        report_lines.append("4. STRESS TEST (30s duration, 5 concurrent users)")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Requests: {stress_metrics['total_requests']}")
        report_lines.append(f"Success Rate: {stress_metrics['success_rate']:.1f}%")
        report_lines.append(f"Requests/Second: {stress_metrics['requests_per_second']:.2f}")
        if 'avg_response_time' in stress_metrics:
            report_lines.append(f"Avg Response Time: {stress_metrics['avg_response_time']:.3f}s")
            report_lines.append(f"P95 Response Time: {stress_metrics['p95_response_time']:.3f}s")
        report_lines.append("")
        
        # Performance recommendations
        report_lines.append("5. PERFORMANCE RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if 'avg_response_time' in load_metrics:
            if load_metrics['avg_response_time'] > 1.0:
                report_lines.append("High response times detected (>1s)")
                report_lines.append("Consider: Model optimization, caching, or scaling")
            else:
                report_lines.append("Response times within acceptable range (<1s)")
        
        if load_metrics['success_rate'] < 95:
            report_lines.append("Low success rate detected (<95%)")
            report_lines.append("Consider: Error handling, timeout adjustments, or scaling")
        else:
            report_lines.append("High success rate achieved (≥95%)")
        
        if stress_metrics['error_rate'] > 5:
            report_lines.append("High error rate under stress (>5%)")
            report_lines.append("Consider: Load balancing, resource limits, or infrastructure scaling")
        else:
            report_lines.append("Low error rate under stress (≤5%)")
        
        # Save report
        report_content = '\n'.join(report_lines)
        with open(save_path, 'w',encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Benchmark report saved to: {save_path}")
        print("\nSummary:")
        print(report_content)
        
        return report_content

if __name__ == '__main__':
    # Run benchmark
    benchmark = APIPerformanceBenchmark()
    benchmark.generate_report()
