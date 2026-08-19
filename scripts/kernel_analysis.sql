.headers on
.mode column

-- This script expects an SQLite file exported from NVIDIA Nsight Systems:
--   sqlite3 report.sqlite < scripts/kernel_analysis.sql

SELECT 'Top kernels by accumulated GPU time' AS section;

SELECT
    s.value AS kernel,
    COUNT(*) AS calls,
    ROUND(SUM(k.end - k.start) / 1000000.0, 2) AS total_ms,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
GROUP BY s.value
ORDER BY SUM(k.end - k.start) DESC
LIMIT 20;

SELECT 'fused_moe_kernel summary' AS section;

SELECT
    COUNT(*) AS calls,
    ROUND(SUM(k.end - k.start) / 1000000.0, 2) AS total_ms,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) AS avg_us,
    ROUND(MIN(k.end - k.start) / 1000.0, 3) AS min_us,
    ROUND(MAX(k.end - k.start) / 1000.0, 3) AS max_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
WHERE s.value LIKE '%fused_moe%';

SELECT 'fused_moe_kernel by block and registers' AS section;

SELECT
    k.blockX,
    k.registersPerThread,
    COUNT(*) AS calls,
    ROUND(SUM(k.end - k.start) / 1000000.0, 2) AS total_ms,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) AS avg_us,
    MIN(k.gridX) AS min_gridX,
    MAX(k.gridX) AS max_gridX
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
WHERE s.value LIKE '%fused_moe%'
GROUP BY k.blockX, k.registersPerThread
ORDER BY calls DESC, k.blockX DESC;

SELECT 'fused_moe_kernel by grid and block' AS section;

SELECT
    k.gridX,
    k.blockX,
    k.registersPerThread,
    COUNT(*) AS calls,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) AS avg_us,
    ROUND(MIN(k.end - k.start) / 1000.0, 3) AS min_us,
    ROUND(MAX(k.end - k.start) / 1000.0, 3) AS max_us,
    ROUND(SUM(k.end - k.start) / 1000000.0, 2) AS total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
WHERE s.value LIKE '%fused_moe%'
GROUP BY k.gridX, k.blockX, k.registersPerThread
ORDER BY calls DESC, total_ms DESC;

SELECT 'NCCL AllReduce summary by exact kernel name' AS section;

SELECT
    s.value AS kernel,
    COUNT(*) AS calls,
    ROUND(SUM(k.end - k.start) / 1000000.0, 2) AS total_ms,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
WHERE s.value LIKE '%AllReduce%'
GROUP BY s.value
ORDER BY SUM(k.end - k.start) DESC;

SELECT 'NCCL AllReduce combined' AS section;

SELECT
    COUNT(*) AS calls,
    ROUND(SUM(k.end - k.start) / 1000000.0, 2) AS total_ms,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
WHERE s.value LIKE '%AllReduce%';

SELECT 'GDN-related kernel name matches' AS section;

SELECT
    s.value AS kernel,
    COUNT(*) AS calls,
    ROUND(SUM(k.end - k.start) / 1000000.0, 2) AS total_ms,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
WHERE lower(s.value) LIKE '%gdn%'
   OR lower(s.value) LIKE '%gated_delta%'
   OR lower(s.value) LIKE '%causal_conv%'
GROUP BY s.value
ORDER BY SUM(k.end - k.start) DESC;

SELECT 'First 200 fused_moe calls in timestamp order' AS section;

SELECT
    k.start,
    ROUND((k.end - k.start) / 1000.0, 3) AS duration_us,
    k.gridX,
    k.blockX,
    k.registersPerThread,
    k.streamId,
    k.deviceId
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.shortName = s.id
WHERE s.value LIKE '%fused_moe%'
ORDER BY k.start
LIMIT 200;
