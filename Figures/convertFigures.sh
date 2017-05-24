for file in ./*
do
filename=$(basename "$file")
extension="${filename##*.}"
body="${filename%.*}"
if  [ "$extension" = "svg" ]
then
  inkscape -D -z --file=$filename --export-pdf=$body.pdf
  echo $filename
fi
done

