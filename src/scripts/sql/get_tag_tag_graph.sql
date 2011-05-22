SELECT COUNT(DISTINCT(A.itemid)) AS weight, B.taginfo_id 
FROM #1 as A, #1 as B 
WHERE A.taginfo_id = #2 
      AND A.itemid = B.itemid 
      AND B.taginfo_id != A.taginfo_id 
      GROUP BY B.taginfo_id
      ORDER BY B.taginfo_id;
