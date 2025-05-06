function DocxHeaders(doc)
  -- Add a page break before each chapter
  for i, section in ipairs(doc.sections) do
    if section.t == "Header" and section.content then
      table.insert(section.content, 1, pandoc.RawBlock("docx", "<w:br w:type='page'/>"))
    end
  end
  return doc
end
