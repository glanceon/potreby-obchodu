# Generated by Django 3.0.8 on 2020-12-27 18:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('market', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Online',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='online/')),
                ('email', models.EmailField(max_length=254)),
            ],
        ),
        migrations.DeleteModel(
            name='myapriori',
        ),
    ]